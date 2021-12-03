import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU

from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from auglichem.utils import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG, NUM_BOND_TYPE, NUM_BOND_DIRECTION



class GINEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        # print(edge_index)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINE(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, 
        task='classification', emb_dim=300, 
        feat_dim=256, num_layer=5, pool='mean', drop_ratio=0, output_dim=None, **kwargs
    ):
        super(GINE, self).__init__()
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
            self.gnns.append(GINEConv(emb_dim, aggr="add"))

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
                nn.Linear(self.feat_dim//2, self.feat_dim//4),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim//4, self.output_dim)
            )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        # h_list = [x]
        # for layer in range(self.num_layer):
        #     h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
        #     h = self.batch_norms[layer](h)
        #     if layer == self.num_layer - 1:
        #         h = F.dropout(h, self.drop_ratio, training = self.training)
        #     else:
        #         h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
        #     h_list.append(h)
        
        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # if layer == self.num_layer - 1:
            #     h = F.dropout(h, self.drop_ratio, training=self.training)
            # else:
            #     h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
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


if __name__ == "__main__":
    model = GINet()
    print(model)
    layer_list = []
    for name, param in model.named_parameters():
        if 'pred_lin' in name:
            print(name, param.requires_grad)
            layer_list.append(name)

    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))
    print(base_params)
    print(params)

    from torch.optim import Adam
    optimizer = Adam([{'params': base_params, 'lr': 1e-4}, {'params': params}], lr=1e-3)
