import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm, Sequential, Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def bootstrap_accuracy(model, data_loader, n_bootstraps=1000, confidence_level=0.95):
    model.eval()
    accuracies = []

    for _ in range(n_bootstraps):
        
        indices = torch.randint(0, len(data_loader.dataset), (len(data_loader.dataset),))
        sample_data = [data_loader.dataset[i] for i in indices]
        sample_loader = DataLoader(sample_data, batch_size=data_loader.batch_size, shuffle=True)
        
        
        acc = test(model, sample_loader)
        accuracies.append(acc)

    
    lower_bound = np.percentile(accuracies, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(accuracies, (1 + confidence_level) / 2 * 100)

    return lower_bound, upper_bound

loss_func = torch.nn.NLLLoss()


def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(data_loader.dataset)


def test(model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        with torch.no_grad():
            out = model(data)
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)

from torch_geometric.data import DataLoader

def augment_data(data_list):
    augmented_data_list = []
    for data in data_list:
        # Randomly drop edges with a probability of 0.3
        if np.random.rand() < 0.3:
            perm = torch.randperm(data.edge_index.size(1))[:int(0.7 * data.edge_index.size(1))]
            data.edge_index = data.edge_index[:, perm]
            data.edge_attr = data.edge_attr[perm]
        # Add noise to node features
        noise = torch.randn_like(data.x) * 0.05
        data.x = data.x + noise
        augmented_data_list.append(data)
    return augmented_data_list


filename = "dataset.csv" 
df = pd.read_csv(filename)

node_encoder = OneHotEncoder(sparse=False)
node_features = node_encoder.fit_transform(df[['src_feat', 'dst_feat']])


edge_encoder = OneHotEncoder(sparse=False)
edge_attributes = edge_encoder.fit_transform(df[['edge_attr']])

df['time_stamp'] = (df['time_stamp'] - df['time_stamp'].mean()) / df['time_stamp'].std()


groups = df.groupby(['session_id', 'graph_id'])


data_list = []

for name, group in groups:
    
    edge_index = torch.tensor(group[['src', 'dst']].to_numpy().T, dtype=torch.long)

    
    node_feat = node_features[group.index]

   
    time_stamp_feat = group['time_stamp'].to_numpy()
    node_feat = np.concatenate([node_feat, time_stamp_feat[:, None]], axis=-1)  # Add new axis to time_stamp
    
    
    edge_attr = edge_attributes[group.index]

    
    label = torch.tensor(group['label'].iloc[0], dtype=torch.long)

    data = Data(x=torch.tensor(node_feat, dtype=torch.float),
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                y=label)
    
    data_list.append(data)


augmented_data_list = augment_data(data_list)

NUM_EDGE_FEATURES = data.num_edge_features

class ImprovedGatModel(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads):
        super(ImprovedGatModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=heads, dropout=0.6, edge_dim=NUM_EDGE_FEATURES)
        self.bn1 = BatchNorm(64 * heads)
        self.conv2 = GATConv(64 * heads, 128, heads=1, concat=True, dropout=0.5, edge_dim=NUM_EDGE_FEATURES)
        self.bn2 = BatchNorm(128)
        self.conv3 = GATConv(128, 128, concat=True, dropout=0.5, edge_dim=NUM_EDGE_FEATURES)
        self.bn3 = BatchNorm(128)
        self.residual_fc = Linear(128, 128)
        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        residual = x  # Residual connection
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.elu(x) + self.residual_fc(residual)  # Adding residual connection
        x = global_mean_pool(x, data.batch)  # Pooling
        return F.log_softmax(self.fc(x), dim=1)


num_node_features = data_list[0].x.shape[1]
model = ImprovedGatModel(num_node_features, 21, heads=8)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)  


train_loader = DataLoader(augmented_data_list, batch_size=32, shuffle=True)
test_loader = DataLoader(data_list, batch_size=32, shuffle=False)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=True)

train_loss, train_acc, test_acc = [], [], []
early_stopping = EarlyStopping(patience=30, min_delta=0.0005)
num_epochs = 300

for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer)
    acc = test(model, train_loader)
    test_acc_epoch = test(model, test_loader)

    train_loss.append(loss)
    train_acc.append(acc)
    test_acc.append(test_acc_epoch)


    scheduler.step()


    if (epoch + 1) % 10 == 0:
        lower_ci, upper_ci = bootstrap_accuracy(model, test_loader)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Test Acc: {test_acc_epoch:.4f}')
        print(f'Test Accuracy Confidence Interval: [{lower_ci:.4f}, {upper_ci:.4f}]')

    if early_stopping(loss):
        print(f"Early stopping at epoch {epoch+1}")
        break