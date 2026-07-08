import torch
import torch.nn as nn

from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool

from sgformer.large.ours import GraphConv
from aegnn.models.layer import MaxPooling

from adaptedsgformer.layers.pooling import Max_voxel_pooling, Avg_voxel_pooling
from adaptedsgformer.layers.trans import TransConv
from adaptedsgformer.utils import compute_pooling_at_each_layer, embed_1D_scalar
from adaptedsgformer.layers.block import BlockDAGT, BlockGT

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

class AEGT(nn.Module):

    def __init__(self,in_channels = 36,
                 out_channels = 2,
                 num_heads = 1,
                 pooling_size = (16,12),
                 input_shape = [120, 100],
                 pe_aggr = "cat", #or "add"
                 max_periods = [120,100,50],
                 factors = [1, 1, 1],
                 dropout_trans = 0.1,
                 dropout_ff = 0.1,
                 dropout_classifier = 0.1,
                 norm_func = 'layer'):

        super(AEGT, self).__init__()

        # assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        
        self.input_shape = torch.tensor(input_shape)

        self.x_embedding = nn.Embedding(embedding_dim=in_channels, num_embeddings=2)

        self.encoding_periods = max_periods
        self.factors = factors

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
        # factors = [1, 1, 1e8]
        embed_pos = torch.stack([
            embed_1D_scalar(batch.pos[:, dim_in] * fact, self.in_channels/3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
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
            embed_1D_scalar(data.pos[:, dim_in] * fact, self.in_channels/3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
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

    def __init__(self, 
                in_channels=24,
                out_channels=2,
                last_voxel_div ='7x5', #voxel division of the last DAGT block
                final_size = 16, # Final size of pooling
                pe_dim=12,
                pe_aggr='cat',
                width = 120,
                height = 100,
                batch_size = 32, 
                pool_aggr = 'max', 
                keep_temporal_ordering=False,
                self_loop=False,
                encoding_periods=[120, 100, 50],
                factors = [1,1,1],
                num_heads = 1,
                dropout_trans = 0.1,
                dropout_ff = 0.1,
                dropout_classifier = 0.1,
                norm_func = 'layer'
                ): 

        super(DAGT, self).__init__()

        assert pe_dim % 3 == 0, f"pe_dim ({pe_dim}) must be divisible by 3."

        self.block_gt_params={
                    "num_heads": num_heads,
                    "dropout_trans": dropout_trans,
                    "dropout_ff": dropout_ff,
                    "norm_func": norm_func,
                }
        
        self.pooling_params = {
                    "width": width,
                    "height": height,
                    "batch_size" : batch_size,
                    "aggr": pool_aggr,
                    "keep_temporal_ordering":keep_temporal_ordering,
                    "self_loop":self_loop,
                }
        
        channels_block = [32, 48, 64, 64, 64]

        sensor_shape = torch.tensor([width, height])

        poolings = compute_pooling_at_each_layer(last_voxel_div, 4)
        voxel_size = sensor_shape / poolings
        
        self.pe_dim = pe_dim
        self.encoding_periods = encoding_periods

        self.x_embedding = nn.Embedding(embedding_dim=in_channels, num_embeddings=2)

        self.pe_aggr = pe_aggr

        self.final_size = final_size

        if self.pe_aggr == 'add':
            pass
        elif self.pe_aggr == 'cat':
            in_channels += pe_dim

        self.blockGT0 = BlockGT(in_channels, channels_block[0], **self.block_gt_params)

        self.blockDAGT1 = BlockDAGT(channels_block[0],
                                    channels_block[1],
                                    voxel_size=voxel_size[0],
                                    pe_dim=pe_dim,
                                    pe_aggr=pe_aggr,
                                    encoding_periods=encoding_periods,
                                    factors= factors,
                                    pooling_params=self.pooling_params,
                                    blockGT_params=self.block_gt_params)
        
        self.blockDAGT2 = BlockDAGT(channels_block[1],
                                    channels_block[2],
                                    voxel_size=voxel_size[1],
                                    pe_dim=pe_dim,
                                    pe_aggr=pe_aggr,
                                    encoding_periods=encoding_periods,
                                    factors= factors,
                                    pooling_params=self.pooling_params,
                                    blockGT_params=self.block_gt_params)
        
        self.blockDAGT3 = BlockDAGT(channels_block[2],
                                    channels_block[3],
                                    voxel_size=voxel_size[2],
                                    pe_dim=pe_dim,
                                    pe_aggr=pe_aggr,
                                    encoding_periods=encoding_periods,
                                    factors= factors,
                                    pooling_params=self.pooling_params,
                                    blockGT_params=self.block_gt_params)
        
        self.blockDAGT4 = BlockDAGT(channels_block[3],
                                    channels_block[4],
                                    voxel_size=voxel_size[3],
                                    pe_dim=pe_dim,
                                    pe_aggr=pe_aggr,
                                    encoding_periods=encoding_periods,
                                    factors= factors,
                                    pooling_params=self.pooling_params,
                                    blockGT_params=self.block_gt_params)
        
        self.final_pooling = Max_voxel_pooling(sensor_shape//4, size=final_size, start = [0., 0.], end=sensor_shape-1)
        
        self.fc = nn.Sequential(nn.Linear(channels_block[4] * final_size, 128, bias=True),
                                nn.GELU(),
                                nn.Dropout(dropout_classifier),
                                nn.Linear(128, out_channels, bias = True)
        )

    
    def forward(self, batch :Batch):

        #Embedding
        embed_pos = torch.stack([
            embed_1D_scalar(batch.pos[:, dim_in] * fact, self.pe_dim//3 ,max_period=max_period) for (dim_in, fact, max_period) in zip(range(3), self.factors, self.encoding_periods)
        ], dim=1)

        embed_pos = embed_pos.reshape(embed_pos.shape[0], -1)

        x_emb = self.x_embedding(batch.x.long()).squeeze(1)

        if self.pe_aggr == 'add':
            batch.x = x_emb + embed_pos
        elif self.pe_aggr == 'cat':
            batch.x = torch.cat((x_emb,embed_pos), dim=1)
        
        batch.x = self.blockGT0(batch.x, batch.batch)

        data = self.blockDAGT1(batch)
        data = self.blockDAGT2(data)
        data = self.blockDAGT3(data)
        data = self.blockDAGT4(data)

        x = self.final_pooling(data.x, data.pos[:, :2], batch = data.batch)

        assert x.size(0) == data.num_graphs * self.final_size
        
        x = x.reshape(data.num_graphs, -1)

        return self.fc(x)