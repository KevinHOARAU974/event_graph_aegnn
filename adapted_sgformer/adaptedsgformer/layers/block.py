import torch
import torch.nn as nn

from torch_geometric.data import Batch

from torch import Tensor

from torch_geometric.nn.norm import BatchNorm, LayerNorm

from adaptedsgformer.layers.pooling import Pooling, Pooling2
from adaptedsgformer.layers.trans import TransConvLayer
from adaptedsgformer.utils import embed_1D_scalar


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

        if in_channels != out_channels:
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.proj = nn.Identity()

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

        x_c = self.proj(x)

        x = self.norm1(x)
        x = self.trans(x, batch)
        x = self.dropout1(x)

        x = x + x_c
        
        x_c = x 

        x = self.norm2(x)
        x = self.ff(x)

        x = x + x_c

        return x
    

class BlockDAGT(nn.Module):

    def __init__(self,
                 in_channels=32,
                 out_channels=32,
                 pe_dim=12,
                 pe_aggr='cat',
                 voxel_size=[1,1],
                 encoding_periods=[120, 100, 50],
                 factors = [1, 1, 1],
                 pooling_params = None,
                 blockGT_params = None,
                 ):
        super(BlockDAGT, self).__init__()

        self.pe_dim = pe_dim

        self.encoding_periods = encoding_periods

        self.pooling = Pooling(voxel_size,
                               in_channels=in_channels,
                               **pooling_params)
        
        self.pe_aggr = pe_aggr

        self.factors = factors

        if self.pe_aggr == "add":
            assert in_channels == pe_dim
            self.in_proj = in_channels
        elif self.pe_aggr == "cat":
            self.in_proj = in_channels + pe_dim
        else:
            raise(f"Invalid aggregation between features and positional encoding: {pe_aggr}")
            
        self.projection = nn.Linear(self.in_proj, self.in_channels, bias=True)

        self.blockGT = BlockGT(
            self.in_channels,
            out_channels,
            **blockGT_params
        )
    
    def forward(self, batch: Batch):

        data = self.pooling(batch)

        # factors = [1, 1, 1e8]
        embed_pos = torch.stack([
            embed_1D_scalar(data.pos[:, dim_in] * fact, self.pe_dim//3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        if self.pe_aggr == "add":
            data.x += embed_pos
        elif self.pe_aggr == "cat":
            data.x = torch.cat((data.x,embed_pos), dim=1)

        data.x = self.projection(data.x)

        data.x = self.blockGT(data.x, data.batch)

        return data


class BlockDectectGT(nn.Module):

    def __init__(self,
                    in_channels=32,
                    out_channels=32,
                    pe_dim=12,
                    pe_aggr='cat',
                    voxel_size=[1,1],
                    encoding_periods=[120, 100, 50],
                    factors = [1, 1, 1],
                    pooling_params = None,
                    blockGT_params = None,
                    ):
        super(BlockDectectGT, self).__init__()

        self.pe_dim = pe_dim #Position encoding dimension

        self.encoding_periods = encoding_periods #Max period for sinusoïdal positional encoding

        self.pooling = Pooling2(voxel_size,
                                in_channels=in_channels,
                                **pooling_params)
        
        self.pe_aggr = pe_aggr #Aggregation of PE and node features

        self.factors = factors #Multiplicative Factors for each channels of positions

        if self.pe_aggr == "add":
            assert in_channels % 3 == 0
            self.pe_dim = in_channels
            self.in_proj = in_channels
        elif self.pe_aggr == "cat":
            assert pe_dim % 3 == 0
            self.in_proj = in_channels + pe_dim
        else:
            raise(f"Invalid aggregation between features and positional encoding: {pe_aggr}")

        self.proj = nn.Linear(self.in_proj, in_channels, bias=True)

        self.blockGT = BlockGT(
            in_channels,
            out_channels,
            **blockGT_params
        )
    
    def forward(self, batch: Batch):

        data = self.pooling(batch)

        embed_pos = torch.stack([
            embed_1D_scalar(data.pos[:, dim_in] * fact, self.pe_dim//3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        if self.pe_aggr == "add":
            data.x += embed_pos
        elif self.pe_aggr == "cat":
            data.x = torch.cat((data.x,embed_pos), dim=1)

        data.x = self.proj(data.x)

        data.x = self.blockGT(data.x, data.batch)

        return data