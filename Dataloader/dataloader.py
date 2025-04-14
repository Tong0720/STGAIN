import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


filename = "dataset.csv"  
df = pd.read_csv(filename)
graph_groups = df.groupby(["session_id", "graph_id"])


def process_graph_data(group):
    unique_nodes = sorted(list(set(group["src"].tolist() + group["dst"].tolist())))
    #print(unique_nodes)
    node_indices = {node: idx for idx, node in enumerate(unique_nodes)}
    #print(node_indices)
    

    time_stamps = group.groupby(["src", "dst"]).first().reset_index()["time_stamp"].values
    #print(time_stamps)
    
    src_features = pd.get_dummies(group.groupby(["src", "dst"]).first().reset_index()["src_feat"]).values
    #print(src_features)
    
    dst_features = pd.get_dummies(group.groupby(["src", "dst"]).first().reset_index()["dst_feat"]).values
    #print(dst_features)
    #src_dst_features = pd.get_dummies(group[["src", "dst"]].applymap(lambda x: node_indices[x])).values
    #print(src_dst_features)
    node_features = np.hstack([src_features, dst_features, time_stamps.reshape(-1, 1)])
    edge_index = group[["src", "dst"]].applymap(lambda x: node_indices[x]).values.T
    edge_attr = pd.get_dummies(group["edge_attr"]).values
    #print(edge_attr)
    label = group["label"].iloc[0]
    #print(edge_attr.shape)
    
    return node_features, edge_index, edge_attr, label



graph_data = [process_graph_data(group) for _, group in graph_groups]
graph_data_train, graph_data_test = train_test_split(graph_data, test_size=0.3, random_state=42)

class CustomGraphDataset(Dataset):
    def __init__(self, graph_data):
        self.graphs = []
        self.x = []
        self.edge_index = []
        self.edge_attr = []
        self.y = []
        for node_features, edge_index, edge_attr, label in graph_data:
            data = Data(x=torch.tensor(node_features, dtype=torch.float32),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                        y=torch.tensor(label, dtype=torch.long),
                        drop_last=True)
            self.graphs.append(data)
            self.graphs.append(data)
            self.edge_attr.append(data.edge_attr)
            self.edge_index.append(data.edge_index)
            self.x.append(data.x)
            self.y.append(data.y)


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.x[idx], self.edge_index[idx], self.edge_attr[idx], self.y[idx]
    

dataset = CustomGraphDataset(graph_data)
train_dataset = CustomGraphDataset(graph_data_train)
test_dataset = CustomGraphDataset(graph_data_test)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,drop_last=True)
