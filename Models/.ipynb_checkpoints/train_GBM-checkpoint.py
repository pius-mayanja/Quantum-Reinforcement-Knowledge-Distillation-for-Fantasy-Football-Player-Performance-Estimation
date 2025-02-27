import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
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

# Define features and target
features = [col for col in data.columns if col not in ['total_points', 'high_performer']]
target = 'high_performer'

X = data[features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "confusion_matrix": conf_matrix,
        "classification_report": classification_report(y_test, y_pred),
        "feature_importances": model.feature_importances_,
    }

# Define the Optuna objective function
def objective(trial):
    # Hyperparameters to optimize
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    
    # Create and train the model
    model = GradientBoostingClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Return validation score (AUC)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)

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
final_model = GradientBoostingClassifier(**study.best_params, random_state=42)
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

# Feature importance plot
feature_importance = metrics["feature_importances"]
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 12))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.title('Feature Importance')
plt.tight_layout()

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

# Prepare metrics for JSON output
metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}

# Save metrics to JSON file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
metrics_path = f"gbm_metrics_{timestamp}.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=4)

print(f"Metrics saved to {metrics_path}")

# Save the model
model_path = f"gbm_model_{timestamp}.pkl"
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