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
        self.W_edge = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, out_channels)
        )
        nn.init.zeros_(self.W_edge[-1].weight)
        nn.init.zeros_(self.W_edge[-1].bias)

    def forward(self, batch):

        N = batch.x.shape[0]
        row, column = batch.edge_index

        d = degree(column, N).float().clamp_min(1)

        distance = torch.norm(batch.edge_attr, dim=-1).clamp_min(1e-6)

        correction = 1 + self.W_edge(batch.edge_attr)

        # print(f'distance: {distance.shape}')
        # print(f"correction: {correction.shape}")
        value = correction / distance.unsqueeze(-1)

        x = self.W(batch.x)

        out = torch.zeros_like(x)
        out.index_add_(0, column, x[row] * value)

        batch.x = out / d.unsqueeze(-1)

        return batch