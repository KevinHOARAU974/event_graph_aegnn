import torch
import torch.nn as nn
import numpy as np
import torch_scatter

from torch_geometric.data import Batch
from typing import List, Optional, Tuple, Union, List, Callable

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, avg_pool_x,voxel_grid
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.pool.avg_pool import _avg_pool_x
from torch_geometric.nn.pool.pool import pool_pos
from torch_geometric.utils import subgraph

from dagr.model.layers.components import BatchNormData

from adaptedsgformer.utils import consecutive_cluster

class Pooling(torch.nn.Module):
    def __init__(self, size: Union[List[float], Tensor], width, height, aggr: str = 'max', keep_temporal_ordering=False, self_loop=False, in_channels=-1):
        super(Pooling, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert aggr in ['mean', 'max']
        self.aggr = aggr
        self.register_buffer("voxel_size", size.to(self.device), persistent=False)

        # self.transform = transform
        self.keep_temporal_ordering = keep_temporal_ordering
        # self.dim = dim

        self.register_buffer("start", torch.Tensor([0,0,0]).to(self.device), persistent=False)
        self.register_buffer("end", torch.Tensor([width-1, height-1]).to(self.device), persistent=False)
        # self.register_buffer("wh_inv", 1/torch.Tensor([[width, height]]), persistent=False)

        # self.max_num_voxels = batch_size * self.num_grid_cells
        # self.register_buffer("sorted_cluster", torch.arange(self.max_num_voxels), persistent=False)

        self.self_loop = self_loop

        self.bn = None
        if in_channels > 0:
            self.bn = LayerNorm(in_channels)

    # @property
    # def num_grid_cells(self):
    #     return (1/self.voxel_size+1e-3).int().prod()
    
    def round_to_pixel(self, pos, wh_inv):
        torch.div(pos+1e-5, wh_inv, out=pos, rounding_mode='floor')
        return pos * wh_inv

    def forward(self, data: Data):


        if data.x.shape[0] == 0:
            return data

        pos = data.pos[:,:2]
        cluster = voxel_grid(pos, batch=data.batch, size=self.voxel_size, start=self.start, end=self.end)
        _, cluster, perm, _ = consecutive_cluster(cluster)
        edge_index = cluster[data.edge_index]
        if self.self_loop:
            edge_index = edge_index.unique(dim=-1)
        else:
            edge_index = edge_index[:, edge_index[0]!=edge_index[1]]
            if edge_index.shape[1] > 0:
                edge_index = edge_index.unique(dim=-1)

        batch = None if data.batch is None else data.batch[perm]
        pos = None if data.pos is None else pool_pos(cluster, data.pos)

        if self.keep_temporal_ordering:
            t_max, _ = torch_scatter.scatter_max(data.pos[:,-1], cluster, dim=0)
            t_src, t_dst = t_max[edge_index]
            edge_index = edge_index[:, t_dst > t_src]

        if self.aggr == 'max':
            x, _ = torch_scatter.scatter_max(data.x, cluster, dim=0)
        else:
            x = _avg_pool_x(cluster, data.x)

        new_data = Batch(batch=batch, x=x, edge_index=edge_index, pos=pos)

        if hasattr(data, "height"):
            new_data.height = data.height
            new_data.width = data.width

        # round x and y coordinates to the center of the voxel grid
        # new_data.pos[:,:2] = self.round_to_pixel(new_data.pos[:,:2], wh_inv=self.wh_inv)

        # if self.transform is not None:
        #     if new_data.edge_index.numel() > 0:
        #         new_data = self.transform(new_data)
        #     else:
        #         new_data.edge_attr = torch.zeros(size=(0,pos.shape[1]), dtype=pos.dtype, device=pos.device)

        if self.bn is not None:
            new_data.x = self.bn(new_data.x)

        return new_data
    
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

class Pooling2(nn.Module):

    def __init__(self, size: List[float], width, height, transform: Callable[[Data, ], Data], aggr: str = 'max', keep_temporal_ordering=False, dim=2, self_loop=False, in_channels=-1, normalisation=False):
        super(Pooling2, self).__init__()
        assert aggr in ['mean', 'max']
        self.aggr = aggr
        self.register_buffer("voxel_size", size, persistent=False)

        self.transform = transform
        self.keep_temporal_ordering = keep_temporal_ordering
        self.dim = dim

        self.register_buffer("start", torch.Tensor([0,0,0]), persistent=False)
        self.register_buffer("end", torch.Tensor([0.9999999,0.9999999,0.9999999]), persistent=False)
        self.register_buffer("wh_inv", 1/torch.Tensor([[width, height]]), persistent=False)

        # self.max_num_voxels = batch_size * self.num_grid_cells
        # self.register_buffer("sorted_cluster", torch.arange(self.max_num_voxels), persistent=False)

        self.self_loop = self_loop

        self.bn = None
        if normalisation:
            self.bn = BatchNormData(in_channels)

    # @property
    # def num_grid_cells(self):
    #     return (1/self.voxel_size+1e-3).int().prod()
    
    def round_to_pixel(self, pos, wh_inv):
        torch.div(pos+1e-5, wh_inv, out=pos, rounding_mode='floor')
        return pos * wh_inv

    def forward(self, data: Data):
        if data.x.shape[0] == 0:
            return data

        pos = torch.cat([data.pos, data.batch.float().view(-1,1)], dim=-1)

        cluster = voxel_grid(pos, batch=data.batch, size=self.voxel_size, start=self.start, end=self.end)

        unique_clusters, cluster, perm, _ = consecutive_cluster(cluster)
        edge_index = cluster[data.edge_index]
        if self.self_loop:
            edge_index = edge_index.unique(dim=-1)
        else:
            edge_index = edge_index[:, edge_index[0]!=edge_index[1]]
            if edge_index.shape[1] > 0:
                edge_index = edge_index.unique(dim=-1)

        batch = None if data.batch is None else data.batch[perm]
        pos = None if data.pos is None else pool_pos(cluster, data.pos)

        if self.keep_temporal_ordering:
            t_max, _ = torch_scatter.scatter_max(data.pos[:,-1], cluster, dim=0)
            t_src, t_dst = t_max[edge_index]
            edge_index = edge_index[:, t_dst > t_src]

        if self.aggr == 'max':
            x, argmax = torch_scatter.scatter_max(data.x, cluster, dim=0)
        else:
            x = _avg_pool_x(cluster, data.x)

        new_data = Batch(batch=batch, x=x, edge_index=edge_index, pos=pos)

        if hasattr(data, "height"):
            new_data.height = data.height
            new_data.width = data.width

        # round x and y coordinates to the center of the voxel grid
        # new_data.pos[:,:2] = self.round_to_pixel(new_data.pos[:,:2], wh_inv=self.wh_inv)

        if self.transform is not None:
            if new_data.edge_index.numel() > 0:
                new_data = self.transform(new_data)
            else:
                new_data.edge_attr = torch.zeros(size=(0,pos.shape[1]), dtype=pos.dtype, device=pos.device)

        if self.bn is not None:
            new_data = self.bn(new_data)

        return new_data



class UniformSampling(nn.Module):

    def __init__(self, n_sample, save_dist, transform=None): # transform just for compatiblity

        super(UniformSampling,self).__init__()

        self.n_sample = n_sample # Number of nodes after sampling
        self.save_dist = save_dist

    def forward(self, batch):

        device = batch.x.device

        batch_list = []

        for b in range(batch.num_graphs):
            graph = batch.get_example(b)
            N = graph.num_nodes
            
            # Random cluster center selection
            chosen_nodes = np.random.choice(np.arange(N), self.n_sample, replace=False)
            chosen_nodes = torch.from_numpy(np.sort(chosen_nodes)).to(device)
            anchor_coords = graph.pos[chosen_nodes]

            # Distance matrix, shape (N x self.n_sample)
            dist_mat = (graph.pos[:, None] - anchor_coords[None]).norm(dim=-1)

            # Calculate node assignment to the closest anchor node
            assignments = torch.argmin(dist_mat, dim=-1)

            # Max pooling of node features
            pooled_features = torch_scatter.scatter(graph.x, assignments, dim=0, dim_size=self.n_sample, reduce='max')

            feature_dict = dict(
                x=pooled_features,
                pos=anchor_coords,
                width=graph.width, 
                height=graph.height, 
                time_window=graph.time_window
            )

            if self.save_dist:
                feature_dict.update({'dist_mat': dist_mat[chosen_nodes]})
            
            new_graph = Data(**feature_dict)
            batch_list.append(new_graph)

        data_batch = Batch.from_data_list(batch_list)

        return data_batch


    # def forward(self, batch: Batch):

    #     idx_sampled = []

    #     for b in range(batch.num_graphs):

    #         idx = torch.where(batch.batch == b)[0]

    #         perm = torch.randperm(idx.numel(), device=idx.device)

    #         idx = idx[perm[:self.n_sample]]

    #         idx_sampled.append(idx) #index of sampled node in this graph

    #     idx_sampled = torch.cat(idx_sampled)



    #     x = batch.x[idx_sampled]
    #     batch_idx = batch.batch[idx_sampled]
    #     pos = batch.pos[idx_sampled]

    #     edge_index, _ = subgraph(idx_sampled, batch.edge_index, relabel_nodes=True, num_nodes=batch.num_nodes)

    #     new_data = Batch(batch=batch_idx, x=x, edge_index=edge_index, pos=pos, width=batch.width, height=batch.height, num_graphs = batch.num_graphs, num_nodes = idx_sampled.numel())

    #     if self.transform is not None:
    #         if new_data.edge_index.numel() > 0:
    #             new_data = self.transform(new_data)
    #         else:
    #             new_data.edge_attr = torch.zeros(size=(0,pos.shape[1]), dtype=pos.dtype, device=pos.device)

    #     return new_data