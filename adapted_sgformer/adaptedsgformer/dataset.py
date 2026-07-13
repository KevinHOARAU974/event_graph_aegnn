import torch

from pathlib import Path

from torch_geometric.data import Dataset
from torch_geometric.transforms import Cartesian


class GraphDataset(Dataset):

    def __init__(self, root, transform = None):
        super().__init__(root, transform=transform)

        self.files = [p for p in Path(root).rglob("*") if p.is_file() and not p.name.startswith(".")]

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data