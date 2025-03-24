import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import optuna
import json
import matplotlib.pyplot as plt
# Function to save results into a JSON file
def save_results_to_json(results, filename="model_results.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

# Sample function to optimize Random Forest
def optimize_random_forest(X_train, y_train):
    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(rf_objective, n_trials=30)
    
    return study.best_params

# Function to evaluate a model and store results
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # Convert to list for JSON serialization
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    return metrics

# Main pipeline
def main():
    # Load and preprocess data (replace this with your actual dataset)
    data = pd.DataFrame({
        'goals_scored': np.random.randint(0, 10, 100),
        'assists': np.random.randint(0, 10, 100),
        'total_points': np.random.randint(0, 200, 100),
        'minutes': np.random.randint(0, 3000, 100),
        'high_performer': np.random.randint(0, 2, 100)  # Target variable
    })
    
    X = data.drop(columns=['high_performer'])
    y = data['high_performer']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimize and train Random Forest
    rf_best_params = optimize_random_forest(X_train, y_train)
    rf_model = RandomForestClassifier(**rf_best_params, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluate model
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")

    # Store results in JSON format
    results = {
        "RandomForest": {
            "best_hyperparameters": rf_best_params,
            "metrics": rf_metrics
        }
    }

    save_results_to_json(results)

# Run the pipeline
if __name__ == "__main__":
    main()

if __name__ == "__main__":
    models_and_metrics = main()
    plt.show()