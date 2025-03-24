import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt

import seaborn as sns
import optuna
import joblib
from datetime import datetime
import json

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

# Select features
selected_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus']
target = 'high_performer'

X = data[selected_features]
y = data[target]

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Feature selection
k_best = SelectKBest(f_classif, k=3)  # Reduce to 3 most important features
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)
selected_feature_indices = k_best.get_support()

print("Selected features:")
for i, selected in enumerate(selected_feature_indices):
    if selected:
        print(f"- {selected_features[i]}")

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
        "classification_report": classification_report(y_test, y_pred)
    }

def objective(trial):
    # Modified hyperparameters with more regularization
    C = trial.suggest_float("C", 0.01, 1.0, log=True)  # Reduced upper bound
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    gamma = trial.suggest_float("gamma", 1e-5, 0.1, log=True)  # Reduced upper bound
    
    # Create SVC model with more regularization
    svc = SVC(
        C=C, 
        kernel=kernel, 
        gamma=gamma, 
        probability=True, 
        random_state=42,
        class_weight='balanced'  # Add class balancing
    )
    
    # Use cross-validation score instead of single split
    scores = cross_val_score(svc, X_train_selected, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Create Optuna study and optimize
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")

try:
    study.optimize(objective, n_trials=50)
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    final_model = SVC(
        C=study.best_params['C'],
        kernel=study.best_params['kernel'],
        gamma=study.best_params['gamma'],
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    
    final_model.fit(X_train_selected, y_train)
    
    # Cross-validation evaluation
    cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    metrics = evaluate_model(final_model, X_test_selected, y_test)
    
    # Print metrics and create visualizations
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
    model_path = f"classical_model_{timestamp}.pkl"
    
    # Prepare metrics for JSON output
    metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}
    
    # Save metrics to JSON file
    metrics_path = f"classical_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    print(f"\nMetrics saved to {metrics_path}")
    
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'selected_features': selected_features,
        'metrics': metrics_json,
        'median_points': median_points,
        'hyperparameters': study.best_params
    }
    
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")
    
    plt.show()

except Exception as e:
    print(f"Error during model training: {e}")
