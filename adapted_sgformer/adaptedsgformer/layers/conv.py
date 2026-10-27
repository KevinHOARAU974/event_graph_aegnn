import torch
import torch.nn as nn

from torch_sparse import SparseTensor, matmul

from torch_geometric.data import Data
from torch_geometric.utils import degree

from dagr.model.layers.spline_conv import MySplineConv

from adaptedsgformer.utils import to_dense

#Spline Conv with dense output of DAGR method, modify to be batch size independant
class SplineConvToDense(MySplineConv):
    def forward(self, data: Data, batch_size: int=None)->torch.Tensor:
        data = super().forward(data)
        if data.batch is None:
            data.batch = torch.zeros(len(data.x), dtype=torch.long, device=data.x.device)
        return self.to_dense(data.x, data.pos, data.pooling, data.batch, batch_size=batch_size)

    def to_dense(self, x, pos, pooling, batch=None, batch_size=None):
        return to_dense(self, x, pos, pooling, batch, batch_size=batch_size)
    
class SmallGraphConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SmallGraphConv,self).__init__()

        self.W = nn.Linear(in_channels, out_channels)
        self.W_edge = nn.Linear(3, 1)

    def forward(self, batch):

        N = batch.x.shape[0]
        row, column = batch.edge_index
        d = degree(column, N).float()
        value = torch.sigmoid(self.W_edge(batch.edge_attr)).squeeze(-1)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=column, col=row, value=value, sparse_sizes=(N,N))
        x = matmul(adj, batch.x) / d.clamp_min(1).unsqueeze(-1)

        batch.x = self.W(x)

        return batch