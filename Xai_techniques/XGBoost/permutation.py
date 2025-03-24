import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.inspection import permutation_importance

# Load dataset
data = pd.read_csv("Models/fpl_players.csv")

# Preprocessing: Drop missing values
data.dropna(inplace=True)

# Encode categorical features
label_encoders = {}
categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Define target variable (binary classification: high vs. low performer)
median_points = data['total_points'].median()
data['high_performer'] = (data['total_points'] >= median_points).astype(int)

# Select features
features = [col for col in data.columns if col not in ['total_points', 'high_performer']]
X = data[features]
y = data['high_performer']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Optuna objective function
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42, use_label_encoder=False)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)

# Hyperparameter tuning with Optuna
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Train final model with best hyperparameters
print("Training final model with best hyperparameters...")
final_model = xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, use_label_encoder=False)
final_model.fit(X_train, y_train)

# Model evaluation
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print performance metrics
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Points', 'High Points'],
            yticklabels=['Low Points', 'High Points'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# 🔍 Permutation Importance (XAI)
perm_importance = permutation_importance(final_model, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()

# Plot Permutation Importance
plt.figure(figsize=(12, 6))
plt.barh([features[i] for i in sorted_idx], perm_importance.importances_mean[sorted_idx], color='royalblue')
plt.xlabel("Mean Accuracy Decrease")
plt.ylabel("Feature")
plt.title("Permutation Importance - Feature Impact on Accuracy")
plt.show()

model_path = f"xgb_model_permutation.pkl"
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': features,
    'hyperparameters': study.best_params,
    'median_points': median_points
}, model_path)

print(f"\nModel saved to {model_path}")
