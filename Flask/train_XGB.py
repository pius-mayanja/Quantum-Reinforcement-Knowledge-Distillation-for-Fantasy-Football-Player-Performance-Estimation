import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

# Load dataset
data = pd.read_csv(r"C:\Users\LENOVO\Downloads\Models\fpl_players.csv")

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

# Create Optuna study and optimize
print("Starting hyperparameter optimization with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Train the final model with best hyperparameters
print("Training final model with best hyperparameters...")
final_model = xgb.XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42, use_label_encoder=False)
final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Evaluate final model
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Points', 'High Points'],
            yticklabels=['Low Points', 'High Points'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Save the model
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = f"xgb_model_{timestamp}.pkl"
joblib.dump({
    'model': final_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': features,
    'hyperparameters': study.best_params,
    'median_points': median_points
}, model_path)

print(f"\nModel saved to {model_path}")


