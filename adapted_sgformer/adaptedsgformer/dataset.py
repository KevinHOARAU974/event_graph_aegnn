import torch

from pathlib import Path

from torch_geometric.data import Dataset


class GraphDataset(Dataset):

    def __init__(self, root):
        super().__init__(root)

        self.files = [p for p in Path(root).rglob("*") if p.is_file()]

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data