import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super(DQN, self).__init__()
        layers = []
        current_dim = input_dim
        
        # Build dynamic hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.fc(x)

# Data Preprocessing
def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    X = data.drop(columns=['total_points'])  # Features
    y = data['total_points']  # Target

    # Label encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load and preprocess data
data = pd.read_csv("fpl_players.csv")
X_train, X_test, y_train, y_test = preprocess_data(data)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create Datasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = 1  # Predicting a continuous value
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Hyperparameters to optimize
    hidden_layers = []
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    for i in range(n_layers):
        hidden_dim = trial.suggest_int(f"hidden_dim_{i}", 32, 256)
        hidden_layers.append(hidden_dim)
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # Recreate data loaders with new batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with trial parameters
    model = DQN(input_dim, hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(10):  # Reduced epochs for optimization
        model.train()
        total_loss = 0
        
        for states, labels in train_loader:
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(states).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for states, labels in valid_loader:
                states, labels = states.to(device), labels.to(device)
                outputs = model(states).squeeze(1)
                valid_loss += criterion(outputs, labels).item()
        
        # Report intermediate value
        trial.report(valid_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return valid_loss

def train_final_model(best_params):
    # Create model with best parameters
    hidden_layers = [best_params[f"hidden_dim_{i}"] 
                    for i in range(best_params["n_layers"])]
    
    model = DQN(input_dim, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()
    
    # Recreate data loader with best batch size
    train_loader = DataLoader(train_dataset, 
                            batch_size=best_params["batch_size"], 
                            shuffle=True)
    
    print("\nTraining final model with best parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Full training with best parameters
    for epoch in range(20):
        model.train()
        total_loss = 0
        for states, labels in train_loader:
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(states).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20, Loss: {total_loss:.4f}")
    
    return model

# Create Optuna study
print("Starting hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("\nBest trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
final_model = train_final_model(study.best_params)

# Save the model
torch.save({
    'model_state_dict': final_model.state_dict(),
    'best_params': study.best_params
}, "dqn_fpl_model_optimized.pth")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for states, labels in test_loader:
            states = states.to(device)
            labels = labels.to(device)

            outputs = model(states).squeeze(1)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert predictions to binary for classification-based metrics
    threshold = 0.5
    all_preds_binary = [1 if pred >= threshold else 0 for pred in all_preds]
    all_labels_binary = [1 if label >= threshold else 0 for label in all_labels]

    # Compute metrics
    accuracy = accuracy_score(all_labels_binary, all_preds_binary)
    precision = precision_score(all_labels_binary, all_preds_binary, zero_division=1)
    recall = recall_score(all_labels_binary, all_preds_binary, zero_division=1)
    f1 = f1_score(all_labels_binary, all_preds_binary, zero_division=1)

    # Compute specificity
    cm = confusion_matrix(all_labels_binary, all_preds_binary)
    tn = cm[0, 0] if cm.shape[0] > 1 else 0
    fp = cm[0, 1] if cm.shape[1] > 1 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "F1 Score": f1,
        "Specificity": specificity
    }

# Evaluate final model
metrics = evaluate_model(final_model, DataLoader(test_dataset, 
                                               batch_size=study.best_params["batch_size"]))
print("\nFinal Model Metrics:", metrics)

# Save metrics to JSON file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
metrics_path = f"DQM_metrics_{timestamp}.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to {metrics_path}")

from matplotlib.pyplot import plot as plt
# Plot optimization history
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.tight_layout()

# Plot parameter importances
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Parameter Importances')
plt.tight_layout()

plt.show()
