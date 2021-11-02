import math

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

from auglichem.utils import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG, NUM_BOND_TYPE, NUM_BOND_DIRECTION


def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.emb_dim = emb_dim
        self.aggr = aggr

        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.bias = Parameter(torch.Tensor(emb_dim))
        self.reset_parameters()

        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, 1)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, 1)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        # glorot(self.weight)
        # zeros(self.bias)
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        edge_index, __ = gcn_norm(edge_index)

        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, edge_attr):
        # return x_j if edge_attr is None else edge_attr.view(-1, 1) * x_j
        return x_j if edge_attr is None else edge_attr + x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)


class GCN(nn.Module):
    def __init__(self, 
        task='classification', emb_dim=300, 
        feat_dim=256, num_layer=5, pool='mean', drop_ratio=0, output_dim=None, **kwargs
    ):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task
        if(output_dim == None):
            self.output_dim = 1
        else:
            self.output_dim = output_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GCNConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if self.task == 'classification':
            self.output_dim *= 2
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.output_dim)
            )
        elif self.task == 'regression':
            self.pred_lin = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim//2), 
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.feat_dim//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//2, self.output_dim)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.feat_lin(h)
        h = self.pool(h, data.batch)

        return h, self.pred_lin(h)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

