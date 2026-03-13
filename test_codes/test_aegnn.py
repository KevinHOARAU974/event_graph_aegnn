import aegnn
import torch
from aegnn.models.networks.graph_res import GraphRes
import numpy as np

tensor_rand = torch.rand([224,224,4])
print(tensor_rand.size())
# print(len)

model = GraphRes("ncaltech101", np.array([224, 224, 2]), 10)
print(model)