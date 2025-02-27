import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
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

# Define features and target
selected_features = ['minutes', 'goals_scored', 'assists', 'clean_sheets', 'bonus', 'influence', 'creativity', 'threat']
X = data[selected_features]
y = data['high_performer']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using L1 regularization
selector = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    max_features=5  # Limit to top 5 features
)
X_selected = selector.fit_transform(X_scaled, y)
selected_feature_mask = selector.get_support()

print("\nSelected features:")
selected_feature_names = [f for f, selected in zip(selected_features, selected_feature_mask) if selected]
print(selected_feature_names)

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Ensure balanced split
)

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
        "coefficients": model.coef_[0],
    }

# Define the Optuna objective function for Logistic Regression
def objective(trial):
    # Modified hyperparameters for better regularization
    params = {
        "C": trial.suggest_float("C", 1e-5, 1e-1, log=True),  # Stronger regularization
        "penalty": trial.suggest_categorical("penalty", ["l2", "elasticnet"]),  # Remove l1 and None
        "solver": trial.suggest_categorical("solver", ["saga"]),  # Limit to saga solver
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }
    
    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    
    # Use cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        model = LogisticRegression(
            **params,
            max_iter=2000,  # Increased max_iter
            random_state=42,
            tol=1e-4  # Increased tolerance
        )
        
        # Use cross-validation with multiple metrics
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        # Combine metrics (emphasize AUC more than accuracy)
        combined_score = 0.3 * accuracy_scores.mean() + 0.7 * auc_scores.mean()
        
        return combined_score
    except Exception as e:
        print(f"Error with parameters {params}: {e}")
        return 0.0

# Create Optuna study and optimize
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print optimization results
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best hyperparameters and early stopping
print("\nTraining final model with best hyperparameters...")
final_model = LogisticRegression(
    **study.best_params,
    max_iter=2000,
    random_state=42,
    tol=1e-4
)

# Perform cross-validation before final training
cv_scores = cross_val_score(final_model, X_train, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train on full training set
final_model.fit(X_train, y_train)

# Calculate training and test scores
train_score = final_model.score(X_train, y_train)
test_score = final_model.score(X_test, y_test)
print(f"\nTraining score: {train_score:.4f}")
print(f"Test score: {test_score:.4f}")
print(f"Difference (train-test): {train_score - test_score:.4f}")

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

# Feature importance (coefficients) plot
coef = metrics["coefficients"]
sorted_idx = np.argsort(np.abs(coef))
plt.figure(figsize=(10, 12))
plt.barh(range(len(sorted_idx)), coef[sorted_idx])
plt.yticks(range(len(sorted_idx)), np.array(selected_feature_names)[sorted_idx])
plt.title('Feature Coefficients')
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

# Save the model
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"logistic_model_{timestamp}.pkl"

# Prepare metrics for JSON output
metrics_json = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items()}

# Save metrics to JSON file
metrics_path = f"Logistic_metrics_{timestamp}.json"
with open(metrics_path, 'w') as f:
    json.dump(metrics_json, f, indent=4)

print(f"Metrics saved to {metrics_path}")

joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': selected_feature_names,
    'hyperparameters': study.best_params,
    'metrics': metrics_json,
    'median_points': median_points
}, model_path)

print(f"\nModel saved to {model_path}")

plt.show()