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


num_node_features = data_list[0].x.shape[1]
model = ImprovedGatModel(num_node_features, 21, heads=8)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)  


train_loader = DataLoader(augmented_data_list, batch_size=32, shuffle=True)



scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5, verbose=True)

train_loss, train_acc= [], []
early_stopping = EarlyStopping(patience=30, min_delta=0.0005)
num_epochs = 300

for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer)
    acc = test(model, train_loader)
    

    train_loss.append(loss)
    train_acc.append(acc)
    


    scheduler.step()


    if (epoch + 1) % 10 == 0:
        lower_ci, upper_ci = bootstrap_accuracy(model, test_loader)
        print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {acc:.4f}')
        

    if early_stopping(loss):
        print(f"Early stopping at epoch {epoch+1}")
        break
