import torch
import torch.nn as nn

import torch_geometric.transforms as T

from torch_geometric.data import Batch

# from dagr.model.networks.net import compute_pooling_at_each_layer
from adaptedsgformer.utils import compute_pooling_at_each_layer

from adaptedsgformer.layers.block import BlockDectectGT, BlockGT
from adaptedsgformer.layers.ev_to_gr import EV_TGN

from adaptedsgformer.utils import embed_1D_scalar, check_graphs

class BackboneGT(nn.Module):

    def __init__(self, 
                    in_channels=24,
                    num_blocks=4,
                    first_trans_block = True,
                    hidden_channels_list=[32, 48, 64, 64, 64],
                    attn_type_block_list=['cat','cat', 'cat', 'cat', 'bias'],
                    pooling_type_list = ['voxel_pooling', 'voxel_pooling', 'voxel_pooling', 'voxel_pooling'],
                    last_voxel_div ='5x7', #voxel division of the last DAGT block
                    final_size = 16, # Final size of pooling
                    pe_dim=12,
                    pe_aggr='cat',
                    width = 120,
                    height = 100,
                    pool_aggr = 'max', 
                    keep_temporal_ordering=False,
                    self_loop=False,
                    encoding_periods=[120, 100, 50],
                    factors = [1,1,1],
                    num_heads = 1,
                    dropout_attn = 0.0,
                    dropout_trans = 0.0,
                    dropout_ff = 0.0,
                    norm_func = 'layer',
                    num_scales = 1,
                    args_gr = None,
                    ): 
    
            super(BackboneGT, self).__init__()
    
            assert pe_dim % 3 == 0, f"pe_dim ({pe_dim}) must be divisible by 3."
            assert len(hidden_channels_list) == num_blocks+1, f'Length of hidden_channels must be num_blocks+1'
            if first_trans_block:
                assert len(attn_type_block_list) == num_blocks+1, f'Length of attn_type_block must be num_blocks+1'
            else:
                assert len(attn_type_block_list) == num_blocks, f'Length of attn_type_block must be num_blocks'
            assert len(pooling_type_list) == num_blocks, f'Length of pooling_type_list must be num_blocks'
    
            self.num_scales = num_scales
    
            self.poolings, self.samplings = compute_pooling_at_each_layer(last_voxel_div, num_layers=num_blocks)
            self.sparse = pooling_type_list[0] == 'uniform_sampling'

            # print(f'poolings: {self.poolings}')
            # print(f'samplings: {self.samplings}')

            max_vals_for_cartesian = 2*self.poolings[:,:2].max(-1).values
            self.strides = torch.ceil(self.poolings[-2:,1] * height).numpy().astype("int32").tolist()
            self.strides = self.strides[-self.num_scales:]
            
            self.encoding_periods = encoding_periods
            self.factors = factors

            self.first_trans_block = first_trans_block
    
            self.x_embedding = nn.Embedding(embedding_dim=in_channels, num_embeddings=2)
    
            self.pe_aggr = pe_aggr
    
            self.final_size = final_size

            self.num_scales = num_scales

            self.events_to_graph = EV_TGN(**args_gr)

            self.hidden_channels_list = hidden_channels_list

            self.cartesian_transform_out = T.Cartesian(norm=True, cat=False, max_value=1.0)
            
            if self.pe_aggr == 'add':
                assert in_channels % 3 == 0
                self.pe_dim = in_channels
                self.in_proj = in_channels
                # self.proj = nn.Identity(self.in_proj)
            elif self.pe_aggr == 'cat':
                assert pe_dim % 3 == 0, f"pe_dim ({pe_dim}) must be divisible by 3."
                self.pe_dim = pe_dim
                self.in_proj = in_channels + pe_dim
                # self.proj = nn.Linear(self.in_proj, in_channels)
            #########
            self.pe_embedding = nn.Sequential(*[
                nn.Linear(self.pe_dim, self.pe_dim),
                nn.LeakyReLU()
            ])
            #########

            if self.first_trans_block:

                if self.pe_aggr == 'add':
                    self.proj = nn.Identity(self.in_proj)
                elif self.pe_aggr == 'cat':
                    self.proj = nn.Linear(self.in_proj, in_channels)

                block_gt_params={
                                        "num_heads": num_heads,
                                        "dropout_trans": dropout_trans,
                                        "dropout_ff": dropout_ff,
                                        "norm_func": norm_func,
                                        "attn_block_type": attn_type_block_list[0],
                                    }

                self.blockGT0 = BlockGT(in_channels, hidden_channels_list[0], **block_gt_params)
                
                attn_type_block_list = attn_type_block_list[1:]
            else:
                self.proj = nn.Linear(self.in_proj, hidden_channels_list[0])
    
            self.num_blocks = num_blocks
            self.block_dagt = nn.ModuleList()
    
            #Num linear attention block
            for i in range(num_blocks):

                block_gt_params = {"num_heads": num_heads,
                                    "dropout_trans": dropout_trans,
                                    "dropout_ff": dropout_ff,
                                    "norm_func": norm_func,
                                    "attn_block_type": attn_type_block_list[i],
                                }
                
                if pooling_type_list[i] == 'voxel_pooling':
                    pooling_params = {  
                                        "size": self.poolings[i],
                                        "in_channels": hidden_channels_list[i],
                                        "width": width,
                                        "height": height,
                                        "aggr": pool_aggr,
                                        "keep_temporal_ordering":keep_temporal_ordering,
                                        "self_loop":self_loop,
                                        "transform": None,
                                    }
                elif pooling_type_list[i] == 'uniform_sampling':
                    pooling_params = {  
                        "n_sample": int(self.samplings[i]),
                        "save_dist": attn_type_block_list[i] == 'bias',
                    }
                    

                if attn_type_block_list[i] == 'bias':
                    block_gt_params["dropout_attn"] = dropout_attn
                    cart = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[i])
                    pooling_params['transform'] = cart

                if i==num_blocks-1:
                    if pooling_type_list[i] == 'voxel_pooling':
                        pooling_params['aggr'] = 'mean'
                    cart = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[i])
                    pooling_params['transform'] = cart                

                self.block_dagt.append(BlockDectectGT(hidden_channels_list[i],
                                                hidden_channels_list[i+1],
                                                pe_dim=self.pe_dim,
                                                pe_aggr=pe_aggr,
                                                encoding_periods=encoding_periods,
                                                factors= factors,
                                                pooling_type= pooling_type_list[i],
                                                pooling_params=pooling_params,
                                                blockGT_params=block_gt_params)
                )

        
    def forward(self, batch :Batch):
        
        device = next(self.parameters()).device
        data = batch.clone().to(device)

        # check_graphs(batch, "DataLoader")
        if not self.sparse:
            data = self.events_to_graph(data)

        # check_graphs(data, "Après ev_to_gr")

        #Embedding

        embed_pos = torch.stack([
            embed_1D_scalar(data.pos[:, dim_in] * fact, self.pe_dim//3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
        ], dim=1)

        embed_pos = self.pe_embedding(embed_pos.reshape(embed_pos.shape[0], -1))

        x_emb = self.x_embedding(data.x.long()).squeeze(1)

        if self.pe_aggr == 'add':
            data.x = x_emb + embed_pos
        elif self.pe_aggr == 'cat':
            data.x = torch.cat((x_emb,embed_pos), dim=1)

        # check_graphs(data, "BACKBONE INPUT")

        data.x = self.proj(data.x)

        if self.first_trans_block:
            data.x = self.blockGT0(data)

        # check_graphs(data, "AFTER BLOCK 0")

        for i in range(self.num_blocks):
            data = self.block_dagt[i](data)
            # check_graphs(data, f"AFTER BLOCK {i+1}")

        if getattr(self.block_dagt[-1].pooling, "voxel_size", None) is not None:
            data.pooling = self.block_dagt[-1].pooling.voxel_size[:3]

        # Post pro for sparse pipeline
        if self.sparse:
            # Construct the graph at the end of the backbone with a maximal radius
            data = self.events_to_graph(data)
            # Remove self loops
            edge_index = data.edge_index
            data.edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            # Cartesian transform
            data = self.cartesian_transform_out(data)

        return data