import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import joblib
from datetime import datetime
import json

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

# Feature Selection - Drop Highly Correlated Features
correlation_matrix = data.corr()
high_corr_features = correlation_matrix[abs(correlation_matrix['total_points']) > 0.8].index.tolist()
high_corr_features.remove('total_points')  # Keep target variable

print(f"Removing highly correlated features: {high_corr_features}")
data.drop(columns=high_corr_features, inplace=True)

# Define features and target
features = [col for col in data.columns if col not in ['total_points', 'high_performer']]
target = 'high_performer'

X = data[features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset (70-30 to allow better generalization)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Create a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # For ROC AUC, we need prediction probabilities
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        # Some SVM configurations might not support predict_proba
        y_pred_proba = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": conf_matrix,
        "classification_report": classification_report(y_test, y_pred)
    }

# Define the Optuna objective function for SVM
def objective(trial):
    # Hyperparameters to optimize
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    # Different hyperparameters depending on the kernel
    if kernel == "linear":
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
        }
    else:
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        
        # Additional params for polynomial kernel
        if kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
    
    # Create and train the model
    model = SVC(**params, probability=True, random_state=42)
    
    try:
        model.fit(X_train, y_train)
        
        # Return validation score (AUC)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_pred_proba)
    except Exception as e:
        print(f"Error with parameters {params}: {e}")
        return 0.0  # Return a poor score for failed configurations

# Create Optuna study and optimize
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print optimization results
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train the final model with best hyperparameters
print("\nTraining final model with best hyperparameters...")
final_model = SVC(**study.best_params, probability=True, random_state=42)
final_model.fit(X_train, y_train)

# Evaluate final model
metrics = evaluate_model(final_model, X_test, y_test)

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

# Plot parallel coordinate plot for parameters
plt.figure(figsize=(12, 6))
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.title('Parallel Coordinate Plot')
plt.tight_layout()

# Save the model
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"SVM_model_{timestamp}.pkl"

# Prepare metrics for JSON output
metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}

# Save metrics to JSON file
metrics_path = f"SVM_metrics_{timestamp}.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=4)

print(f"Metrics saved to {metrics_path}")

joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': features,
    'hyperparameters': study.best_params,
    'metrics': metrics_json,
    'median_points': median_points
}, model_path)

print(f"\nModel saved to {model_path}")

plt.show()