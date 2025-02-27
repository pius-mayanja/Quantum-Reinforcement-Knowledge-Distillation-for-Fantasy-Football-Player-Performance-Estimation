import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import json
from datetime import datetime
import optuna
from optuna.trial import Trial

# Load dataset
df = pd.read_csv("fpl_players.csv")

# Feature selection
features = [
    "goals_scored", "assists", "minutes", "goals_conceded",
    "creativity", "influence", "threat", "bonus", "bps", "ict_index",
    "clean_sheets", "red_cards", "yellow_cards", "selected_by_percent", "now_cost"
]
target = "total_points"

# Feature engineering
# Add interaction terms for key features
df['goal_assist_interaction'] = df['goals_scored'] * df['assists']
df['minutes_influence'] = df['minutes'] * df['influence'] / 100
features.extend(['goal_assist_interaction', 'minutes_influence'])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target].values

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Split dataset - fixed to handle duplicate values
# Simple train-test split first
train_size = int(0.8 * len(df))
test_size = len(df) - train_size

# Try to create stratified split, but fall back to random if necessary
try:
    # Create bins for stratification with the duplicates parameter
    df['points_bin'] = pd.qcut(df[target], q=5, labels=False, duplicates='drop')
    train_indices = []
    test_indices = []

    # Stratified split
    for bin_value in df['points_bin'].unique():
        bin_indices = np.where(df['points_bin'] == bin_value)[0]
        np.random.shuffle(bin_indices)
        split_idx = int(0.8 * len(bin_indices))
        train_indices.extend(bin_indices[:split_idx])
        test_indices.extend(bin_indices[split_idx:])
except Exception as e:
    print(f"Falling back to random split due to: {str(e)}")
    # Fall back to random split if stratification fails
    indices = list(range(len(df)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

# Create datasets
train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
test_dataset = TensorDataset(X_tensor[test_indices], y_tensor[test_indices])

# Define improved model architecture
class ImprovedPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(ImprovedPredictionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers with batch normalization and dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.hidden_layers(x)
        return self.output_layer(x)

# Define the objective function for Optuna
def objective(trial: Trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    hidden_layer_1 = trial.suggest_categorical('hidden_layer_1', [128, 256, 512])
    hidden_layer_2 = trial.suggest_categorical('hidden_layer_2', [64, 128, 256])
    hidden_layer_3 = trial.suggest_categorical('hidden_layer_3', [32, 64, 128])
    
    # Define batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedPredictionModel(
        input_dim=X.shape[1],
        hidden_dims=[hidden_layer_1, hidden_layer_2, hidden_layer_3],
        dropout_rate=dropout_rate
    ).to(device)
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Loss function - Huber loss is more robust to outliers
    criterion = nn.HuberLoss(delta=1.0)
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 7
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Evaluate the model
    mae, mse, r2 = evaluate(model, test_loader, device)
    
    # Return the metric to optimize
    return mae

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predicted_points = model(X_batch)
            
            all_preds.extend(predicted_points.cpu().numpy().flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())
    
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    return mae, mse, r2

def main():
    # Run the Optuna study with more trials for better results
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    # Print the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    # Train the final model with the best hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = ImprovedPredictionModel(
        input_dim=X.shape[1],
        hidden_dims=[best_params['hidden_layer_1'], best_params['hidden_layer_2'], best_params['hidden_layer_3']],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    # Define data loaders with the best batch size
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer with the best learning rate and weight decay
    optimizer = optim.AdamW(
        final_model.parameters(), 
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Loss function
    criterion = nn.HuberLoss(delta=1.0)
    
    # Training the final model
    epochs = 100  # Train for longer with the best hyperparameters
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 10  # More patience for the final model
    
    for epoch in range(epochs):
        # Training phase
        final_model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        final_model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = final_model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save(final_model.state_dict(), 'best_fpl_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    final_model.load_state_dict(torch.load('best_fpl_model.pth'))
    
    # Evaluate the final model
    mae, mse, r2 = evaluate(final_model, test_loader, device)
    
    # Save metrics to JSON file
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "R^2": float(r2),
        "Best Hyperparameters": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in best_params.items()}
    }
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = f"fpl_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Save important feature information
    final_model.eval()
    # Create a dummy input to get feature importance
    dummy_input = torch.ones((1, X.shape[1]), dtype=torch.float32).to(device)
    dummy_input.requires_grad = True
    
    # Forward pass
    output = final_model(dummy_input)
    output.backward()
    
    # Get gradients
    gradients = dummy_input.grad.cpu().numpy()[0]
    feature_importance = np.abs(gradients)
    
    # Normalize importances
    feature_importance = feature_importance / np.sum(feature_importance)
    
    # Create a dictionary of feature importances
    feature_names = features
    importance_dict = {feature: float(importance) for feature, importance in zip(feature_names, feature_importance)}
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    
    # Save feature importance
    with open(f"feature_importance_{timestamp}.json", 'w') as f:
        json.dump(sorted_importance, f, indent=4)

if __name__ == "__main__":
    main()