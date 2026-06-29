import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import List, Optional, Tuple, Union, List

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, avg_pool_x,voxel_grid
from torch_geometric.nn.norm import BatchNorm, LayerNorm



from sgformer.large.ours import GraphConv
from aegnn.models.layer import MaxPooling


def embed_1D_scalar(t, dim, max_period):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

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
        # qs = qs / torch.norm(qs, p=2)  # [B, Nmax, H, M]
        # ks = ks / torch.norm(ks, p=2)  # [B, Nmax, H, M]

        qs = F.normalize(qs, p=2, dim=-1, eps=1e-6)
        ks = F.normalize(ks, p=2, dim=-1, eps=1e-6)

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

class Max_voxel_pooling(nn.Module):

    def __init__(self, voxel_size: List[int], size: int, start: Optional[Union[float, List[float], Tensor]] = None, end: Optional[Union[float, List[float], Tensor]] = None):

        super(Max_voxel_pooling, self).__init__()
        self.voxel_size = voxel_size
        self.size = size
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        
        pos = pos.float()

        if batch is not None:
            batch = batch.long()

        if torch.is_tensor(self.voxel_size):
            self.voxel_size = self.voxel_size.to(device=pos.device, dtype=pos.dtype)
        
        if torch.is_tensor(self.start):
            self.start = self.start.to(device=pos.device, dtype=pos.dtype)

        if torch.is_tensor(self.end):
            self.end = self.end.to(device=pos.device, dtype=pos.dtype)
        
        # print(f"device end: {self.end}")
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size, start=self.start, end=self.end)

        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"
    
class Avg_voxel_pooling(nn.Module):

    def __init__(self, voxel_size: List[int], size: int, start: Optional[Union[float, List[float], Tensor]] = None, end: Optional[Union[float, List[float], Tensor]] = None):

        super(Avg_voxel_pooling, self).__init__()
        self.voxel_size = voxel_size
        self.size = size
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        
        pos = pos.float()

        if batch is not None:
            batch = batch.long()

        if torch.is_tensor(self.voxel_size):
            self.voxel_size = self.voxel_size.to(device=pos.device, dtype=pos.dtype)
        
        if torch.is_tensor(self.start):
            self.start = self.start.to(device=pos.device, dtype=pos.dtype)

        if torch.is_tensor(self.end):
            self.end = self.end.to(device=pos.device, dtype=pos.dtype)
        
        # print(f"device end: {self.end}")
        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size, start=self.start, end=self.end)

        x, _ = avg_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"

class AdaptedSGFormer(nn.Module):
    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 pe_dim = 10, #dim for one component of pe
                 embedding_pe_aggr = 'add',
                 trans_num_layers=1,
                 trans_num_heads=1,
                 trans_dropout=0.5,
                 trans_use_bn=True,
                 trans_use_residual=True,
                 trans_use_weight=True,
                 trans_use_act=True,
                 gnn_num_layers=1,
                 gnn_dropout=0.5,
                 gnn_use_weight=True,
                 gnn_use_init=False,
                 gnn_use_bn=True,
                 gnn_use_residual=True,
                 gnn_use_act=True,
                 linear_dim=128,
                 linear_dropout=0.1,
                 use_graph=True,
                 graph_weight=0.8,
                 encoding_periods=None,
                 aggregate='add',
                 pooling_type='global', #or 'voxel'
                 pooling_function = 'mean',
                 sensor_size = [120, 100],
                 voxel_div = 4 #Division of the sensor size
                 ): 
        
        super().__init__()

        self.embedding_pe_aggr = embedding_pe_aggr

        if embedding_pe_aggr == 'add':
            layer_in = in_channels
            self.pe_dim = in_channels
        elif embedding_pe_aggr == 'cat':
            self.pe_dim = 3 * pe_dim
            layer_in = in_channels + self.pe_dim
        else:
            raise ValueError(f'Invalid embedding pe aggregation :{embedding_pe_aggr}')

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        print(f'in_channels:{self.in_channels}')

        #Global self attention path
        self.trans_conv = TransConv(layer_in, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)

        #GNN path
        self.graph_conv = GraphConv(layer_in, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)

        self.use_graph = use_graph #If the GNN path is use or not
        self.graph_weight = graph_weight #Convex add

        self.aggregate = aggregate #aggregation function
        self.pooling_type = pooling_type
        self.pooling_function = pooling_function #pooling function

        self.h_map = None

        self.encoding_periods = encoding_periods
        self.x_embedding = nn.Embedding(embedding_dim=self.in_channels, num_embeddings=2)

        self.sensor_size = torch.tensor(sensor_size)
        self.voxel_div = voxel_div
        #Pooling layer

        if pooling_type == 'global':
            if pooling_function == 'mean':
                self.pooling_layer = global_mean_pool
            elif pooling_function == 'max':
                self.pooling_layer = global_max_pool
            else:
                raise ValueError(f'Invalid pooling function:{pooling_function}')
            
        elif pooling_type == 'voxel':
            if pooling_function == 'mean':
                self.pooling_layer = Avg_voxel_pooling(self.sensor_size // voxel_div, size=voxel_div**2, start=[0., 0.], end=self.sensor_size - 1)
            elif pooling_function == 'max':
                self.pooling_layer = Max_voxel_pooling(self.sensor_size // voxel_div, size=voxel_div**2, start=[0., 0.], end=self.sensor_size - 1)
            else:
                raise ValueError(f'Invalid pooling function:{pooling_function}')
            
        else:
                raise ValueError(f'Invalid pooling type:{pooling_type}')
        
        #Classifier head

        if pooling_type == 'global':
            if aggregate == 'add':
                self.fc = self.fc = nn.Sequential(
                    nn.Linear(hidden_channels, linear_dim),
                    nn.ReLU(),
                    nn.Dropout(linear_dropout),
                    nn.Linear(linear_dim, out_channels),
                    )
            elif aggregate == 'cat':
                self.fc = nn.Sequential(
                    nn.Linear(2 * hidden_channels, linear_dim),
                    nn.ReLU(),
                    nn.Dropout(linear_dropout),
                    nn.Linear(linear_dim, out_channels),
                    )
            else:
                raise ValueError(f'Invalid aggregate type:{aggregate}')
            
        elif pooling_type == 'voxel':
            if aggregate == 'add':
                self.fc = nn.Sequential(
                    nn.Linear(hidden_channels * self.voxel_div**2, out_channels)
                )
            elif aggregate == 'cat':
                self.fc = nn.Sequential(
                    nn.Linear(2 * hidden_channels * self.voxel_div**2, out_channels)
                )

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, batch : Batch):

        #Embedding
        factors = [1, 1, 1e8]
        embed_pos = torch.stack([
            embed_1D_scalar(batch.pos[:, dim_in] * fact, self.pe_dim/3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        x_emb = self.x_embedding(batch.x.long()).squeeze()
        
        # Aggregation embedding features and positionnal encoding
        if self.embedding_pe_aggr == 'add':
            batch.x = x_emb + embed_pos
        elif self.embedding_pe_aggr == 'cat':
            batch.x = torch.cat((x_emb, embed_pos), dim=1)

        # print(batch)
        x1 = self.trans_conv(batch)
        if self.use_graph:
            x2 = self.graph_conv(batch.x, batch.edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1


        if self.pooling_type == 'global':
            x = self.pooling_layer(x, batch.batch) 
        if self.pooling_type == 'voxel':
            x = self.pooling_layer(x, batch.pos[:,:2], batch=batch.batch)
            x = x.reshape(batch.num_graphs, -1)

        self.h_map = x
        x = self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


class BlockGT(nn.Module):

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 dropout_trans = 0.1,
                 dropout_ff = 0.1,
                 norm_func = 'layer',
                 ):
        super(BlockGT, self).__init__()

        if norm_func == 'layer':
            norm = LayerNorm
        elif norm_func == 'batch':
            norm = BatchNorm

        self.norm1 = norm(in_channels) 
        self.trans = TransConvLayer(in_channels, out_channels, num_heads)
        self.dropout1 = nn.Dropout(dropout_trans)

        self.norm2 = norm(out_channels)
        self.ff = nn.Sequential(
            nn.Linear(out_channels, 4*out_channels, bias=True),
            nn.Dropout(dropout_ff),
            nn.GELU(),
            nn.Linear(4*out_channels, out_channels,bias=True),
            nn.Dropout(dropout_ff)
        )


    def forward(self, x: Tensor, batch: Tensor):

        x_c = x

        x = self.norm1(x)
        x = self.trans(x, batch)
        x = self.dropout1(x)

        x = x + x_c
        
        x_c = x 

        x = self.norm2(x)
        x = self.ff(x)

        x = x + x_c

        return x

class AEGT(nn.Module):

    def __init__(self,in_channels = 36,
                 out_channels = 2,
                 num_heads = 1,
                 pooling_size = (16,12),
                 input_shape = [120, 100],
                 pe_aggr = "cat", #or "add"
                 max_periods = [120,100,50],
                 dropout_trans = 0.1,
                 dropout_ff = 0.1,
                 dropout_classifier = 0.1,
                 norm_func = 'layer'):

        super(AEGT, self).__init__()

        # assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        
        self.input_shape = torch.tensor(input_shape)

        self.x_embedding = nn.Embedding(embedding_dim=in_channels, num_embeddings=2)

        self.encoding_periods = max_periods

        self.in_channels = in_channels

        self.pe_aggr = pe_aggr

        if pe_aggr == 'cat':
            self.proj_1 = nn.Linear(2 * self.in_channels, self.in_channels, bias = True)
            self.proj_2 = nn.Linear(2 * self.in_channels, self.in_channels, bias = True)
        elif pe_aggr == 'add':
            pass
        else:
            raise(f"Invalide aggregation type : {pe_aggr}")

        self.block1 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
        self.block2 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
        self.block3 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
        self.block4 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
        self.block5 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)

        self.pool5 = MaxPooling(pooling_size, start = [0., 0.], end= self.input_shape-1)

        self.block6 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
        self.block7 = BlockGT(in_channels, in_channels, num_heads, dropout_trans=dropout_trans, dropout_ff=dropout_ff, norm_func=norm_func)
    
        self.pool7 = Max_voxel_pooling(self.input_shape//4, size=16, start = [0., 0.], end= self.input_shape-1)
        self.fc = nn.Sequential(nn.Linear(in_channels * 16, 128, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout_classifier),
                                nn.Linear(128, out_channels, bias = True)
        )


    def forward(self, batch : Batch):

        #Embedding
        factors = [1, 1, 1e8]
        embed_pos = torch.stack([
            embed_1D_scalar(batch.pos[:, dim_in] * fact, self.in_channels/3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        x_emb = self.x_embedding(batch.x.long()).squeeze()

        if self.pe_aggr == 'add':
            x = x_emb + embed_pos
        elif self.pe_aggr == 'cat':
            x = torch.cat((x_emb,embed_pos), dim=1)
            x = self.proj_1(x)


        x = self.block1(x, batch.batch)
        x = self.block2(x, batch.batch)

        x_c = x.clone()

        x = self.block3(x, batch.batch)
        x = self.block4(x, batch.batch)

        x = x + x_c

        x = self.block5(x, batch.batch)
        
        data = self.pool5(x, pos=batch.pos, batch=batch.batch, edge_index=batch.edge_index, return_data_obj=True)
        
        #Reinject positional encoding in features after pooling
        embed_pos = torch.stack([
            embed_1D_scalar(data.pos[:, dim_in] * fact, self.in_channels/3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        if self.pe_aggr == 'add':
            x = data.x + embed_pos
        elif self.pe_aggr == 'cat':
            x = torch.cat((data.x,embed_pos), dim=1)
            x = self.proj_2(x)

        x_c = x.clone()

        x = self.block6(x, data.batch)
        x = self.block7(x, data.batch)

        x = x + x_c

        x = self.pool7(x, pos = data.pos[:, :2], batch = data.batch)

        x = x.reshape(data.num_graphs, -1)

        return self.fc(x)
    


class DAGT(nn.Module):
     pass