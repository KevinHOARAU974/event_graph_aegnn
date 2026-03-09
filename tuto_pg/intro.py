import os
import torch
import torch.nn as nn

# %matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def visualize_graph(G, color):

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False, node_color=color, cmap="Set2")
    plt.show()

def visualize_embedding(h, color, epoch=None, loss=None):

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


#On utilise ici un dataset de graphe
dataset = KarateClub()
print(f"Dataset: {dataset}:")
print("===================")
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')


data = dataset[0]

print(data)
print('=====================')

#Quelques caractéristiques sur le graphe
print(f"Number of nodes: {data.num_nodes}")
print(f'Number of edges: {data.num_edges}')
print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
print(f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}") # Indique le pourcentage de train dans le graphe
print(f"Has isolate nodes: {data.has_isolated_nodes()}")
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

#Visualisation des edges [noeud de départ, noeud d'arrivée]
edge_index = data.edge_index
print(edge_index.t())

#Visualisation d'un graphe
G = to_networkx(data, to_undirected=True)
visualize_graph(G, color = data.y)

class GCN(nn.Module):

    def __init__(self):

        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = nn.Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):

        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh() #Dernière GNN embedding space

        out = self.classifier(h)

        return out, h
    
model = GCN()
print(model)

_, h = model(data.x, edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize_embedding(h, color=data.y)