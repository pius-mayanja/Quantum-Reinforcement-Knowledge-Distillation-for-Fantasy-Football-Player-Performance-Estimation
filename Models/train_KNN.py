import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, BatchNorm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from sklearn.neighbors import kneighbors_graph
import optuna
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import json

# Load data
df = pd.read_csv('fpl_players.csv')

# Encode categorical variables
label_encoder = LabelEncoder()
df["element_type"] = label_encoder.fit_transform(df["element_type"])

# Feature engineering (adding interaction terms)
df["goal_assist_interaction"] = df["goals_scored"] * df["assists"]
selected_features = ["element_type", "minutes", "goals_scored", "assists", "clean_sheets", "bonus", "ict_index", "goal_assist_interaction"]
features = df[selected_features]
target = np.log1p(df["total_points"].values)  # Apply log transform to target

# Define optuna objective function for GAT model hyperparameters
def objective(trial):
    # Hyperparameters to tune
    hidden_channels = trial.suggest_int("hidden_channels", 32, 256, log=True)
    heads = trial.suggest_int("heads", 1, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    k_neighbors = trial.suggest_int("k_neighbors", 3, 10)
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Construct a k-nearest neighbors graph
    num_players = len(df)
    adj_matrix = kneighbors_graph(features_scaled, k_neighbors, mode='connectivity', include_self=True)
    edge_index = np.array(adj_matrix.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # Convert features and target to tensors
    x = torch.tensor(features_scaled, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    
    # Define GAT Model with tunable parameters
    class GATModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout):
            super(GATModel, self).__init__()
            self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.bn1 = BatchNorm(hidden_channels * heads)
            self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
            self.bn2 = BatchNorm(hidden_channels * heads)
            self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
            self.relu = nn.ReLU()
            self.output_activation = nn.ReLU()  # Ensure non-negative output
    
        def forward(self, x, edge_index):
            x = self.gat1(x, edge_index)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.gat2(x, edge_index)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.gat3(x, edge_index)
            x = self.output_activation(x)  # Apply non-negative activation
            return x
    
    # Initialize model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATModel(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Custom loss function: MSE
    loss_fn = nn.MSELoss()
    
    def clip_gradients(model, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Train-test split
    train_mask, test_mask = train_test_split(range(num_players), test_size=0.2, random_state=42)
    train_mask = torch.tensor(train_mask, dtype=torch.long)
    test_mask = torch.tensor(test_mask, dtype=torch.long)
    
    # Move data to device
    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    
    # Training loop with early stopping
    epochs = 100  # Reduced max epochs
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x, edge_index).squeeze()
        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        clip_gradients(model)  # Apply gradient clipping
        optimizer.step()
        
        # Check for early stopping
        val_loss = loss_fn(out[test_mask], y[test_mask]).item()
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index).squeeze()
    
    # Compute metrics
    y_true = np.expm1(y[test_mask].cpu().numpy())  # Reverse log transform
    y_pred = np.expm1(predictions[test_mask].cpu().numpy())
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negativity
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Return RMSE (negative, as Optuna minimizes the objective by default)
    return r2  # Maximize R²

# Run Optuna optimization
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")  # Maximize R²
study.optimize(objective, n_trials=30)

# Print optimization results
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best hyperparameters
print("\nTraining final model with best hyperparameters...")

# Extract best parameters
hidden_channels = study.best_params["hidden_channels"]
heads = study.best_params["heads"]
dropout = study.best_params["dropout"]
learning_rate = study.best_params["learning_rate"]
weight_decay = study.best_params["weight_decay"]
k_neighbors = study.best_params["k_neighbors"]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Construct a k-nearest neighbors graph
num_players = len(df)
adj_matrix = kneighbors_graph(features_scaled, k_neighbors, mode='connectivity', include_self=True)
edge_index = np.array(adj_matrix.nonzero())
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Convert features and target to tensors
x = torch.tensor(features_scaled, dtype=torch.float)
y = torch.tensor(target, dtype=torch.float)

graph_data = Data(x=x, edge_index=edge_index, y=y)

# Define GAT Model with best hyperparameters
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=heads, dropout=dropout):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * heads)
        self.gat3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
        self.relu = nn.ReLU()
        self.output_activation = nn.ReLU()  # Ensure non-negative output

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.gat3(x, edge_index)
        x = self.output_activation(x)  # Apply non-negative activation
        return x

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATModel(in_channels=x.shape[1], hidden_channels=hidden_channels, out_channels=1).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Custom loss function: MSE
loss_fn = nn.MSELoss()

def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# Train-test split
train_mask, test_mask = train_test_split(range(num_players), test_size=0.2, random_state=42)
train_mask = torch.tensor(train_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)

# Move data to device
x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)

# Training loop
epochs = 100
patience = 10
best_loss = float('inf')
patience_counter = 0

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(x, edge_index).squeeze()
    loss = loss_fn(out[train_mask], y[train_mask])
    loss.backward()
    clip_gradients(model)  # Apply gradient clipping
    optimizer.step()

    # Check for early stopping
    val_loss = loss_fn(out[test_mask], y[test_mask]).item()
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Evaluate model
model.eval()
with torch.no_grad():
    predictions = model(x, edge_index).squeeze()

# Compute metrics
y_true = np.expm1(y[test_mask].cpu().numpy())  # Reverse log transform
y_pred = np.expm1(predictions[test_mask].cpu().numpy())
y_pred = np.maximum(y_pred, 0)  # Ensure non-negativity

# Calculate metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"Final Model R²: {r2:.4f}")
print(f"Final Model RMSE: {rmse:.4f}")
print(f"Final Model MAE: {mae:.4f}")

# Prepare metrics for JSON output
metrics = {
    "R2": r2,
    "RMSE": rmse,
    "MAE": mae
}

# Save metrics to JSON file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
metrics_path = f"KNN_metrics_{timestamp}.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to {metrics_path}")

# Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"gat_model_{timestamp}.pth"
model_path = "KNN_model_{timestamp}.pkl"
torch.save(model.state_dict(), model_path)
print(f"Trained model saved to {model_path}")
