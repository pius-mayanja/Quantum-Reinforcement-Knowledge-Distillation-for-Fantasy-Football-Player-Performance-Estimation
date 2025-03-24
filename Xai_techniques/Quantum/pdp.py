import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from datetime import datetime
import json

# Try to import pennylane - if it fails, provide installation instructions
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
except ImportError as e:
    print("PennyLane packages are required for this script.")
    print("Please ensure PennyLane is installed correctly in your Python environment.")
    print("You can try installing them using: pip install pennylane")
    print(f"The specific import error was: {e}")
    exit(1)

# Set random seed
np.random.seed(42)

# Load dataset
data = pd.read_csv("Models/fpl_players.csv")

# Preprocessing
data.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Convert total_points into binary classification (above/below median)
median_points = data['total_points'].median()
data['high_performer'] = (data['total_points'] >= median_points).astype(int)

# Due to quantum computing limitations, we need to select a subset of important features
selected_features = ['bps', 'ict_index','minutes']
target = 'high_performer'

X = data[selected_features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample a smaller dataset due to quantum simulation limitations
sample_size = min(70, len(X_scaled))
indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sampled = X_scaled[indices]
y_sampled = y.iloc[indices].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)

# Define the QuantumClassifier class
class QuantumClassifier:
    def __init__(self, feature_dim, n_layers, learning_rate=0.01, n_epochs=30):
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.quantum_model = self._create_quantum_model()
        
    def _create_quantum_model(self):
        dev = qml.device("default.qubit", wires=self.feature_dim)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs, weights):
            AngleEmbedding(inputs, wires=range(self.feature_dim))
            StronglyEntanglingLayers(weights, wires=range(self.feature_dim))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.feature_dim)]
        
        return quantum_circuit
    
    def _compute_prediction(self, x, weights):
        quantum_output = self.quantum_model(x, weights)
        return float(np.mean(quantum_output) > 0)
    
    def fit(self, X, y):
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.feature_dim)
        self.weights = np.random.randn(*weight_shape)
        
        optimizer = qml.GradientDescentOptimizer(self.learning_rate)
        
        def cost_fn(weights):
            predictions = np.array([self._compute_prediction(x, weights) for x in X])
            return np.mean((predictions - y) ** 2)
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.weights = optimizer.step(cost_fn, self.weights)
            if epoch % 5 == 0:
                cost = cost_fn(self.weights)
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be trained before prediction")
        return np.array([self._compute_prediction(x, self.weights) for x in X])

# Function to compute PDP for a single feature
def plot_pdp(model, X_train, feature_idx, feature_name, ax, n_points=50):
    """
    Plot Partial Dependence Plot (PDP) for a given feature.

    Parameters:
    model: Trained model (QuantumClassifier)
    X_train: The training data
    feature_idx: The index of the feature to analyze
    feature_name: The name of the feature to analyze
    ax: The axis to plot on (for grid layout)
    n_points: The number of points to sample for PDP plot
    """
    
    # Create an array of values for the feature of interest
    feature_values = np.linspace(X_train[:, feature_idx].min(), X_train[:, feature_idx].max(), n_points)
    
    # Create an array to hold the predicted values for each feature value
    predictions = np.zeros(n_points)
    
    # For each value in the feature_values array, calculate the prediction
    for i, value in enumerate(feature_values):
        X_temp = X_train.copy()
        X_temp[:, feature_idx] = value  # Set the feature of interest to the current value
        predictions[i] = np.mean(model.predict(X_temp))  # Average over the predictions
    
    # Plotting the PDP
    ax.plot(feature_values, predictions, color='blue')
    ax.set_title(f'PDP for {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Average Prediction')
    ax.grid(True)

# Train the model
final_model = QuantumClassifier(feature_dim=X_train.shape[1], n_layers=2, learning_rate=0.01, n_epochs=30)
final_model.fit(X_train, y_train)

# Set up grid for PDP plots
features = ['bps', 'ict_index', 'minutes']
n_features = len(features)

# Create a grid of subplots
fig, axes = plt.subplots(1, n_features, figsize=(15, 5))

# Plot PDPs for each feature
for i, feature in enumerate(features):
    plot_pdp(final_model, X_train, feature_idx=i, feature_name=feature, ax=axes[i])

# Adjust layout and display
plt.tight_layout()
plt.show()
