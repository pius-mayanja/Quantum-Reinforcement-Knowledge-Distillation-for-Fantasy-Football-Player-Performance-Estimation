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
data = pd.read_csv("fpl_players.csv")

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
# Let's choose a few features most likely to impact performance
selected_features = ['minutes', 'goals_scored', 'bonus']
target = 'high_performer'

X = data[selected_features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample a smaller dataset due to quantum simulation limitations
# For real-world quantum computation, adjust based on your quantum resource availability
sample_size = min(70, len(X_scaled))
indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sampled = X_scaled[indices]
y_sampled = y.iloc[indices].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.3, random_state=42)

# Create a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    try:
        # Some quantum models might support predict_proba
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        # Fall back to decision function or binary predictions
        try:
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
        except:
            # If no scoring is available, use the binary predictions
            auc = roc_auc_score(y_test, y_pred)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": conf_matrix,
        "classification_report": classification_report(y_test, y_pred)
    }

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

# Define the Optuna objective function
def objective(trial):
    n_layers = trial.suggest_int("n_layers", 1, 2)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    
    try:
        feature_dim = X_train.shape[1]
        model = QuantumClassifier(
            feature_dim=feature_dim,
            n_layers=n_layers,
            learning_rate=learning_rate,
            n_epochs=n_epochs
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    except Exception as e:
        print(f"Error with parameters: {e}")
        return 0.0

# Create Optuna study and optimize (with limited trials due to quantum simulation cost)
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")

try:
    study.optimize(objective, n_trials=10)  # Reduced trials due to computational intensity
    
    # Print optimization results
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train the final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    
    final_model = QuantumClassifier(
        feature_dim=X_train.shape[1],
        n_layers=study.best_params['n_layers'],
        learning_rate=study.best_params['learning_rate'],
        n_epochs=study.best_params['n_epochs']
    )
    
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
    
    print(f"\nFinal Model Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    
    # Plotting confusion matrix
    plt.figure(figsize=(8, 6))
    cm = metrics["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Points', 'High Points'],
                yticklabels=['Low Points', 'High Points'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    plt.tight_layout()
    
    # Plot parameter importance
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    plt.tight_layout()
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = f"Quantum_model_{timestamp}.pkl"
    
    # Prepare metrics for JSON output
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
    
    # Save metrics to JSON file
    metrics_path = f"Quantum_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)

    print(f"\nMetrics saved to {metrics_path}")
    
    # Save the model configuration
    joblib.dump({
        'model_type': 'PennyLane_Quantum_Classifier',
        'hyperparameters': study.best_params,
        'weights': final_model.weights,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'selected_features': selected_features,
        'metrics': metrics_json,
        'median_points': median_points
    }, model_path)
    
    print(f"\nModel configuration saved to {model_path}")
    
    plt.show()

except Exception as e:
    print(f"Error during quantum model training: {e}")
    print("Quantum computing requires specific hardware or simulators.")
    print("You may need to adjust parameters based on available resources.")