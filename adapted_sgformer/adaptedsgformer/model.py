import torch
import torch.nn as nn
import pandas as pd

from pathlib import Path

from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from torch import Tensor
import torch.nn.functional as F

from sgformer.large.ours import GraphConv

# Adapted self Attention layer of SGFormer
class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, input : Tensor, batch : Tensor, output_attn=False):
        # feature transformation

        #B : Batch size
        #Nmax : number of nodes of the largest graph in the batch
        #H : Number of Heads
        #I : Input size
        #M : Output size

        # Groupe by graph in order to have global attention by graph in batch
        x, mask_dense = to_dense_batch(input, batch) #[B, Nmax, I]
        
        batch_size = len(batch.unique())

        qs = self.Wq(x).reshape(batch_size, -1, self.num_heads, self.out_channels) 
        ks = self.Wk(x).reshape(batch_size, -1, self.num_heads, self.out_channels)

        if self.use_weight:
            vs = self.Wv(x).reshape(batch_size, -1, self.num_heads, self.out_channels)
        else:
            vs = x.reshape(batch_size, -1, 1, self.out_channels)

        #Set to zeros padding elements in order that they not contribute to attention
        qs[~mask_dense] = 0.0 
        ks[~mask_dense] = 0.0
        vs[~mask_dense] = 0.0

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [B, Nmax, H, M]
        ks = ks / torch.norm(ks, p=2)  # [B, Nmax, H, M]
        N = mask_dense.sum(dim=1) # [B] Number of nodes in each graph 

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, Nmax, H, D]
        attention_num += N.view(batch_size, 1, 1, 1) * vs

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, Nmax, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [B, Nmax, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N.view(batch_size, 1, 1, 1)
        attn_output = attention_num / attention_normalizer  # [B,Nmax, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [Nmax, Nmax]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [Nmax,1]
            attention = attention / normalizer
        
        attn_output = attn_output[mask_dense]

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output



class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, batch : Batch):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](batch.x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, batch.batch)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class AdaptedSGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True, gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add', pooling = 'mean'):
        
        super().__init__()

        #Global self attention path
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)

        #GNN path
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)


        self.use_graph = use_graph #If the GNN path is use or not
        self.graph_weight = graph_weight #Convex add

        self.aggregate = aggregate #aggregation function
        self.pooling = pooling #pooling function
        
        #Pooling layer
        if pooling == 'mean':
            self.pooling_layer = global_mean_pool
        elif pooling == 'add':
            self.pooling_layer = global_add_pool
        elif pooling == 'max':
            self.pooling_layer = global_max_pool
        else:
            raise ValueError(f'Invalid pooling type:{pooling}')
        
        #Classifier head
        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, batch : Batch):
        x1 = self.trans_conv(batch)
        if self.use_graph:
            x2 = self.graph_conv(batch.x, batch.edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.pooling_layer(x, batch.batch) 
        x = self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()