import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
# from torch_geometric.nn import GINEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


ATOM_LIST = list(range(1,100))
num_atom_type = len(ATOM_LIST) + 1 # including the extra mask tokens

num_bond_type = 2 # self-connected and otherwise


class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.Softplus(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        # self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.ones(x.size(0)) * (num_bond_type-1) # bond type for self-loop edge
        # self_loop_attr[:,0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        edge_embeddings = self.edge_embedding1(edge_attr)

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        feat_dim (int): feature dimenstion
        drop_ratio (float): dropout rate
        pool (str): pooling function
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=256, feat_dim=512, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.num_atom_type = num_atom_type
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Linear(3, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        # nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError('Pooling must be either mean, max, or add')

        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim//2), 
            nn.Softplus(),
            nn.Linear(self.feat_dim//2, 1)
        )

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        h = self.x_embedding1(data.atomics) + self.x_embedding2(data.pos)
        # h = self.x_embedding1(data.atomics) + \
        #     self.x_embedding2(data.pos) + \
        #     self.x_embedding3(data.feat)

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.softplus(h), self.drop_ratio, training=self.training)

        feat = self.feat_lin(h)
        feat = self.pool(feat, data.batch)
        out = self.head(feat)

        self.feat = feat
        
        return out


if __name__ == "__main__":
    model = GINet()
    print(model)
