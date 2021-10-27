from typing import Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter, GRUCell

from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros

#from data_aug.dataset import ATOM_LIST, CHIRALITY_LIST, BOND_LIST, BONDDIR_LIST
from auglichem.utils import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG, NUM_BOND_TYPE, NUM_BOND_DIRECTION

#NUM_ATOM_TYPE = len(ATOM_LIST) + 1 # including the extra mask tokens
#NUM_CHIRALITY_TAG = len(CHIRALITY_LIST)
#NUM_BOND_TYPE = len(BOND_LIST) + 1 # including aromatic and self-loop edge
#NUM_BOND_DIRECTION = len(BONDDIR_LIST)


class GATEConv(MessagePassing):
    def __init__(self, emb_dim, dropout=0.0):
        super(GATEConv, self).__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.att_l = Parameter(torch.Tensor(1, emb_dim))
        self.att_r = Parameter(torch.Tensor(1, emb_dim))

        self.lin1 = Linear(emb_dim*2, emb_dim, False)
        self.lin2 = Linear(emb_dim, emb_dim, False)

        self.bias = Parameter(torch.Tensor(emb_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        out += self.bias
        return out

    def message(self, x_j, x_i, edge_attr,
                index, ptr, size_i: Optional[int]):
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(nn.Module):
    """
    Args:
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(self, task, emb_dim=300, num_layers=5, num_timesteps=3, drop_ratio=0,
                 output_dim=None, **kwargs):
        super(AttentiveFP, self).__init__()
        if(output_dim == None):
            self.output_dim = 1
        else:
            self.output_dim = output_dim


        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.task = task
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.drop_ratio = drop_ratio

        self.lin1 = Linear(emb_dim, emb_dim)

        conv = GATEConv(emb_dim, drop_ratio)
        gru = GRUCell(emb_dim, emb_dim)
        self.atom_convs = torch.nn.ModuleList([conv])
        self.atom_grus = torch.nn.ModuleList([gru])
        for _ in range(num_layers - 1):
            conv = GATConv(emb_dim, emb_dim, dropout=drop_ratio,
                           add_self_loops=False, )#negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(emb_dim, emb_dim))

        self.mol_conv = GATConv(emb_dim, emb_dim,
                                dropout=drop_ratio, add_self_loops=False,
                                )#negative_slope=0.01)
        self.mol_gru = GRUCell(emb_dim, emb_dim)

        # self.lin2 = Linear(emb_dim, out_channels)
        if self.task == 'classification':
            self.output_dim *= 2
            self.lin2 = nn.Linear(emb_dim, self.output_dim)
        elif self.task == 'regression':
            self.lin2 = nn.Linear(emb_dim, self.output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        """"""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.atom_convs[0](x, edge_index, edge_attr))
        h = F.dropout(h, p=self.drop_ratio, training=self.training)
        x = self.atom_grus[0](h, x).relu_()

        for conv, gru in zip(self.atom_convs[1:], self.atom_grus[1:]):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(data.batch.size(0), device=data.batch.device)
        edge_index = torch.stack([row, data.batch], dim=0)

        out = global_add_pool(x, data.batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.drop_ratio, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.drop_ratio, training=self.training)
        return out, self.lin2(out)
