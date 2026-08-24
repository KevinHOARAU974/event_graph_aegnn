import torch

from torch_geometric.data import Data
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