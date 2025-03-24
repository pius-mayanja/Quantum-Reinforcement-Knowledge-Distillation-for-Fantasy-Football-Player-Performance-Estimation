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

# Quantum computing imports
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
except ImportError as e:
    print("PennyLane packages are required for this script.")
    print("Please install using: pip install pennylane")
    print(f"Error: {e}")
    exit(1)

# Explainability imports
try:
    from lime import lime_tabular
except ImportError as e:
    print("\nLIME package required for explanations.")
    print("Install using: pip install lime")
    exit(1)

# Set random seed
np.random.seed(42)

# Load and preprocess data
data = pd.read_csv("Models/fpl_players.csv")
data.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Create binary target
median_points = data['total_points'].median()
data['high_performer'] = (data['total_points'] >= median_points).astype(int)

# Feature selection and preprocessing
selected_features = ['bps', 'ict_index', 'minutes']
X = data[selected_features]
y = data['high_performer']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduced dataset for quantum simulation
sample_size = min(70, len(X_scaled))
indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sampled = X_scaled[indices]
y_sampled = y.iloc[indices].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)

# Modified Quantum Classifier with continuous outputs
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
    
    def _compute_raw_score(self, x, weights):
        return np.mean(self.quantum_model(x, weights))
    
    def fit(self, X, y):
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_layers, n_wires=self.feature_dim)
        self.weights = np.random.randn(*weight_shape)
        
        optimizer = qml.GradientDescentOptimizer(self.learning_rate)
        
        def cost_fn(weights):
            raw_scores = np.array([self._compute_raw_score(x, weights) for x in X])
            return np.mean((raw_scores - y) ** 2)
        
        for epoch in range(self.n_epochs):
            self.weights = optimizer.step(cost_fn, self.weights)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Cost = {cost_fn(self.weights):.4f}")
    
    def predict(self, X):
        return (self.predict_raw(X) > 0).astype(int)
    
    def predict_raw(self, X):
        if self.weights is None:
            raise ValueError("Model not trained")
        return np.array([self._compute_raw_score(x, self.weights) for x in X])

# Optuna optimization
def objective(trial):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_epochs': trial.suggest_int('n_epochs', 5, 20)
    }
    
    model = QuantumClassifier(
        feature_dim=X_train.shape[1],
        **params
    )
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

# Main execution
if __name__ == "__main__":
    # Hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # Train final model
    final_model = QuantumClassifier(
        feature_dim=X_train.shape[1],
        **study.best_params
    )
    final_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = final_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # LIME Explanations
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=selected_features,
        class_names=['Low', 'High'],
        mode='regression',
        verbose=True,
        discretize_continuous=False
    )

    # Explain first test instance
    exp = explainer.explain_instance(
        X_test[0],
        final_model.predict_raw,
        num_features=len(selected_features)
    )

    # Visualization
    plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title("LIME Explanation for Quantum Model Prediction")
    plt.show()

    # Show original feature values
    instance_unscaled = scaler.inverse_transform(X_test[0].reshape(1, -1))
    print("\nOriginal feature values:")
    for feat, value in zip(selected_features, instance_unscaled[0]):
        print(f"{feat}: {value:.2f}")

    