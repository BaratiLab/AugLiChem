from typing import Optional, List, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Sequential, Linear, ReLU, Dropout
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d

import torch_sparse
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_softmax
from torch_scatter import scatter
from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing, DeepGCNLayer
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

from auglichem.utils import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG, NUM_BOND_TYPE, NUM_BOND_DIRECTION


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super(MLP, self).__init__(*m)


class GENConv(MessagePassing):
    def __init__(self, emb_dim: int, 
                 aggr: str = 'softmax', t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False, msg_norm: bool = False,
                 learn_msg_scale: bool = False, norm: str = 'batch',
                 num_layer: int = 2, eps: float = 1e-7):
        
        super(GENConv, self).__init__()

        self.emb_dim = emb_dim
        self.aggr = aggr
        self.eps = eps

        assert aggr in ['softmax', 'softmax_sg', 'power', 'add', 'mean', 'max']

        channels = [emb_dim]
        for i in range(num_layer - 1):
            channels.append(emb_dim * 2)
        channels.append(emb_dim)
        self.mlp = MLP(channels, norm=norm)

        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.initial_t = t
        self.initial_p = p

        if learn_t and aggr == 'softmax':
            self.t = Parameter(torch.Tensor([t]), requires_grad=True)
        else:
            self.t = t

        if learn_p:
            self.p = Parameter(torch.Tensor([p]), requires_grad=True)
        else:
            self.p = p
        
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def reset_parameters(self):
        reset(self.mlp)
        if self.msg_norm is not None:
            self.msg_norm.reset_parameters()
        if self.t and isinstance(self.t, Tensor):
            self.t.data.fill_(self.initial_t)
        if self.p and isinstance(self.p, Tensor):
            self.p.data.fill_(self.initial_p)

    def forward(self, x, edge_index, edge_attr, size=None):

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
        # print(edge_index)

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings, size=size)

        if self.msg_norm is not None:
            out = self.msg_norm(x[0], out)

        x_r = x[1]
        if x_r is not None:
            out += x_r

        return self.mlp(out)

    def message(self, x_j, edge_attr):
        msg = x_j if edge_attr is None else x_j + edge_attr
        return F.relu(msg) + self.eps

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        if self.aggr == 'softmax':
            out = scatter_softmax(inputs * self.t, index, dim=self.node_dim)
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'softmax_sg':
            out = scatter_softmax(inputs * self.t, index,
                                  dim=self.node_dim).detach()
            return scatter(inputs * out, index, dim=self.node_dim,
                           dim_size=dim_size, reduce='sum')

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1 / self.p)

        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)


class DeepGCN(nn.Module):
    def __init__(self, 
        task='classification', num_layer=28, emb_dim=300, 
        feat_dim=256, pool='mean', drop_ratio=0, output_dim=None, **kwargs
    ):
        """
          Inputs:
          -------
          emb_dim (int): Edge feature dimensionality.
          aggr (str, optional, default='softmax'): Aggregate function, one of 'softmax',
                                  'softmax_sg', 'power', 'add', 'mean', 'max'.
          t (float optional, default=1.0): Scaling parameter for softmax and
                                  softmax_sg aggregation.
          learn_t (bool optional, default=False): Flag to learn t or not.
          p (float, optional, default=1.0): Power used for power aggreagation.
          learn_p (bool, optional, default=False): Flag to learn p or not.
          msg_norm (bool, optional, default=False): Flag to normalize messages or not.
          learn_msg_scale (bool, optional, default=False): Flag to learn message norm or not.
          norm (str, optional, default ='batch'): Type of norm to use in MLP.
                                  One of 'batch', 'layer',     or 'instance'.
          num_layer (int, optional, default=2): Number of layers in the network.
          eps (float, optional, default=1e-7): Small value to add to message output.

          Outputs:
          --------
          None
        """
        super(DeepGCN, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio
        self.task = task
        if(output_dim == None):
            self.output_dim = 1
        else:
            self.output_dim = output_dim

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for i in range(1, num_layer + 1):
            conv = GENConv(emb_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layer=2, norm='layer')
            norm = LayerNorm(emb_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.gnns.append(layer)

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

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h = self.gnns[0].conv(h, edge_index, edge_attr)
        for layer in self.gnns[1:]:
            h = layer(h, edge_index, edge_attr)
            h = F.dropout(h, self.drop_ratio, training=self.training)
        h = self.gnns[0].act(self.gnns[0].norm(h))

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
