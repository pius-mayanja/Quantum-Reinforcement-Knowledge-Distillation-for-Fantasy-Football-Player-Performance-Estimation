import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import random
import time
import optuna
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    mean_absolute_error,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)


# Enhanced XGBoost Model for Player Performance Prediction
class XGBoostModel:
    def __init__(self, df, features, target, classification=False):
        self.df = df
        self.features = features
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.classification = classification
        self.training_time = 0
        self.metrics = {}
        self.best_params = {}
        
    def preprocess_data(self):
        X = self.df[self.features]
        y = self.df[self.target]
        
        # If classification, convert target to binary
        if self.classification:
            # Convert to binary classification (above median = 1, below = 0)
            threshold = y.median()
            y = (y > threshold).astype(int)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def optimize_hyperparameters(self, n_trials=100):
        """Use Optuna to find the best hyperparameters"""
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data()
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'random_state': 42
            }
            
            if self.classification:
                param['objective'] = 'binary:logistic'
                param['eval_metric'] = 'logloss'
                model = xgb.XGBClassifier(**param)
            else:
                param['objective'] = 'reg:squarederror'
                param['eval_metric'] = 'rmse'
                model = xgb.XGBRegressor(**param)
            
            # Train with early stopping
            eval_set = [(X_test_scaled, y_test)]
            model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                verbose=False
                #callbacks=[xgb.callback.EarlyStopping(rounds=50)]
            )
            
            # Calculate error
            preds = model.predict(X_test_scaled)
            if self.classification:
                error = 1.0 - accuracy_score(y_test, preds)
            else:
                error = np.sqrt(mean_squared_error(y_test, preds))
                
            return error
        
        # Create a study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        
        # Visualize optimization results
        fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Optimization History')
        plt.savefig('optimization_history.png')
        
        fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title('Parameter Importances')
        plt.savefig('param_importances.png')
        
        print(f"Best hyperparameters: {self.best_params}")
        return study.best_params
        
    def train_model(self, epochs=100, use_optuna=True, n_trials=50):
        """Train the XGBoost model with epochs and optional hyperparameter optimization"""
        X_train_scaled, X_test_scaled, y_train, y_test = self.preprocess_data()
        
        # Optimize hyperparameters if requested
        if use_optuna:
            print("Optimizing hyperparameters with Optuna...")
            best_params = self.optimize_hyperparameters(n_trials)
        else:
            # Default parameters
            if self.classification:
                best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'random_state': 42
                }
            else:
                best_params = {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'random_state': 42
                }
        
        # Set objective and eval metric based on task type
        if self.classification:
            best_params['objective'] = 'binary:logistic'
            best_params['eval_metric'] = 'logloss'
            self.model = xgb.XGBClassifier(**best_params)
        else:
            best_params['objective'] = 'reg:squarederror'
            best_params['eval_metric'] = 'rmse'
            self.model = xgb.XGBRegressor(**best_params)
        
        # Create evaluation set for early stopping
        eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]
        
        # Train model with epochs and measure time
        start_time = time.time()
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=eval_set,
            verbose=True,
            #callbacks=[xgb.callback.EarlyStopping(rounds=20)]
        )
        self.training_time = time.time() - start_time
        
        # Evaluate the model
        self.evaluate_model(X_test_scaled, y_test)
        
        print(f"Model trained in {self.training_time:.2f} seconds")
        return self.model
    
    def evaluate_model(self, X_test_scaled, y_test):
        """Evaluate the model and store metrics"""
        # Make predictions
        if self.classification:
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate classification metrics
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
        else:
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate regression metrics
            self.metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        # Print metrics
        print("\nModel Evaluation:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Visualize results
        if not self.classification:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs Predicted Values')
            plt.savefig('prediction_results.png')
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions with the trained model"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, filename='xgboost_model.pkl'):
        """Save the model to a file"""
        model_info = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'target': self.target,
            'metrics': self.metrics,
            'best_params': self.best_params,
            'training_time': self.training_time,
            'classification': self.classification
        }
        joblib.dump(model_info, filename)
        print(f"Model saved to {filename}")


# Enhanced RL Agent for Team Selection
class TeamSelectionAgent:
    def __init__(self, df, budget, positions, xgb_model, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.df = df
        self.budget = budget
        self.positions = positions  # Positions required for GK, DEF, MID, FW
        self.xgb_model = xgb_model  # XGBoost model for player performance prediction
        self.epsilon = epsilon  # Exploration probability
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = defaultdict(lambda: np.zeros(len(df)))
        self.team_history = []
        self.metrics = {}
    
    def select_action(self, state):
        """
        Epsilon-greedy strategy to choose an action (player) based on the current state.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.df)))  # Explore: Random player selection
        else:
            return np.argmax(self.q_table[state])  # Exploit: Select best player
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning update rule.
        """
        best_future_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * best_future_q - self.q_table[state][action])
    
    def train(self, episodes=100):
        """Train the agent for a number of episodes"""
        start_time = time.time()
        rewards_history = []
        
        for episode in range(episodes):
            if episode % 10 == 0:
                print(f"Training episode {episode}/{episodes}")
            
            state = tuple([0] * len(self.df))  # Start with an empty team
            selected_players = []
            current_budget = self.budget
            positions_filled = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0, "Unknown": 0}
            episode_reward = 0
            
            # Select players to form a team
            while len(selected_players) < sum(self.positions.values()):
                action = self.select_action(state)
                player = self.df.iloc[action]
                
                # Get player position
                position = player["element_type"]
                position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
                pos = position_map.get(position, "Unknown")
                
                try:
                    # Check if we need more players in this position
                    if pos in positions_filled and positions_filled[pos] >= self.positions.get(pos, 0):
                        continue  # Skip this player, position already filled
                    
                    # Check if player can be selected within the budget and not already selected
                    if player["now_cost"] <= current_budget and state[action] == 0:
                        selected_players.append(player)
                        current_budget -= player["now_cost"]
                        positions_filled[pos] += 1
                        
                        # Update state (mark the player as selected)
                        new_state = list(state)
                        new_state[action] = 1
                        next_state = tuple(new_state)
                        
                        # Use the XGBoost model to predict the player's points
                        player_features = player[self.xgb_model.features].values.reshape(1, -1)
                        player_df = pd.DataFrame([player[self.xgb_model.features]])
                        predicted_points = self.xgb_model.predict(player_df)[0]
                        
                        # Reward based on predicted performance and cost efficiency
                        reward = predicted_points / (player["now_cost"] / 10)  # Value for money
                        episode_reward += reward
                        
                        # Update Q-value
                        self.update_q_value(state, action, reward, next_state)
                        state = next_state

                except Exception as e:
                    print(pos)
                    print(player["now_cost"])
                    print(current_budget)
                    print(state[action])

                    print("finally")
                    print(e)

                    exit()
            
            rewards_history.append(episode_reward)
            
            # Decay epsilon (explore less over time)
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # Store team history
            if episode % 10 == 0 or episode == episodes - 1:
                self.team_history.append({
                    'episode': episode,
                    'team': selected_players,
                    'total_cost': self.budget - current_budget,
                    'predicted_points': sum([self.xgb_model.predict(pd.DataFrame([p[self.xgb_model.features]]))[0] for p in selected_players])
                })
        
        training_time = time.time() - start_time
        print(f"Agent training completed in {training_time:.2f} seconds")
        
        # Plot rewards history
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Agent Training Progress')
        plt.savefig('agent_training_progress.png')
        
        self.metrics = {
            'training_time': training_time,
            'episodes': episodes,
            'final_epsilon': self.epsilon,
            'final_reward': rewards_history[-1] if rewards_history else 0
        }
        
        return rewards_history
    
    def get_team(self):
        """Generate the best team based on the trained agent"""
        state = tuple([0] * len(self.df))  # Start with an empty team
        selected_players = []
        current_budget = self.budget
        positions_filled = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        
        # Set epsilon to 0 for pure exploitation
        saved_epsilon = self.epsilon
        self.epsilon = 0
        
        # Keep track of players we've tried but couldn't select
        attempted_players = set()
        
        while len(selected_players) < sum(self.positions.values()):
            # Get best action based on Q-values
            action = self.select_action(state)
            
            # If we've already tried this player, choose next best
            if action in attempted_players:
                # Temporarily set this action's Q-value to a very low value
                original_q = self.q_table[state][action]
                self.q_table[state][action] = -1000
                action = self.select_action(state)
                # Restore the original Q-value
                self.q_table[state][action] = original_q
            
            player = self.df.iloc[action]
            
            # Get player position
            position = player["element_type"]
            position_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
            pos = position_map.get(position, "Unknown")
            
            # Check if position requirements are met
            if pos in positions_filled and positions_filled[pos] >= self.positions.get(pos, 0):
                attempted_players.add(action)
                continue
            
            # Check if player can be selected within budget
            if player["now_cost"] <= current_budget and state[action] == 0:
                selected_players.append(player)
                current_budget -= player["now_cost"]
                positions_filled[pos] += 1
                
                # Update state
                new_state = list(state)
                new_state[action] = 1
                state = tuple(new_state)
            else:
                attempted_players.add(action)
        
        # Restore epsilon
        self.epsilon = saved_epsilon
        
        # Calculate team metrics
        total_cost = self.budget - current_budget
        predicted_points = sum([
            self.xgb_model.predict(pd.DataFrame([p[self.xgb_model.features]]))[0] 
            for p in selected_players
        ])
        
        print(f"Team selected with total cost: {total_cost/10} out of {self.budget/10}")
        print(f"Predicted total points: {predicted_points:.2f}")
        
        return selected_players


# Main execution
if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Models\fpl_players.csv")
    
    # Define features for the model
    features = ["goals_scored", "assists", "minutes", "ict_index", "bonus", 
                "bps", "influence", "creativity", "threat", "selected"]
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    print(f"Available features: {available_features}")
    
    # Define target for the model
    target = "total_points"
    
    # Create a metrics table to store results of different models
    metrics_table = pd.DataFrame(columns=["Model", "accuracy", "auc", "Precision", 
                                         "Recall (Sensitivity)", "F1 Score", 
                                         "MAE", "MSE", "R2", "RMSE", "Training Time (s)"])
    
    # Train regression model
    print("\n--- Training XGBoost Regression Model ---")
    regression_model = XGBoostModel(df, available_features, target, classification=False)
    regression_model.train_model(epochs=50, use_optuna=True, n_trials=20)
    regression_model.save_model("xgboost_regression_model.pkl")
    
    # Add regression metrics to the table
    new_row = {
        "Model": "XGBoost Regression",
        "MAE": regression_model.metrics.get('mae', '-'),
        "MSE": regression_model.metrics.get('mse', '-'),
        "R2": regression_model.metrics.get('r2', '-'),
        "RMSE": regression_model.metrics.get('rmse', '-'),
        "Training Time (s)": regression_model.training_time
    }
    metrics_table = pd.concat([metrics_table, pd.DataFrame([new_row])], ignore_index=True)
    
    # Train classification model
    print("\n--- Training XGBoost Classification Model ---")
    classification_model = XGBoostModel(df, available_features, target, classification=True)
    classification_model.train_model(epochs=50, use_optuna=True, n_trials=20)
    classification_model.save_model("xgboost_classification_model.pkl")
    
    # Add classification metrics to the table
    new_row = {
        "Model": "XGBoost Classification",
        "accuracy": classification_model.metrics.get('accuracy', '-'),
        "auc": classification_model.metrics.get('auc', '-'),
        "Precision": classification_model.metrics.get('precision', '-'),
        "Recall (Sensitivity)": classification_model.metrics.get('recall', '-'),
        "F1 Score": classification_model.metrics.get('f1_score', '-'),
        "Training Time (s)": classification_model.training_time
    }
    metrics_table = pd.concat([metrics_table, pd.DataFrame([new_row])], ignore_index=True)
    
    # Initialize RL Agent with the regression model
    print("\n--- Training Team Selection Agent ---")
    agent = TeamSelectionAgent(
        df, 
        budget=1000,  # Budget in price units (actual budget = 100.0)
        positions={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}, 
        xgb_model=regression_model
    )
    
    # Train the agent
    rewards = agent.train(episodes=200)
    
    # Get the final team
    print("\n--- Selected Team ---")
    selected_team = agent.get_team()
    selected_team_df = pd.DataFrame(selected_team)
    
    # Save the team to CSV
    selected_columns = [
        "first_name", "second_name", "team", "element_type",
        "now_cost", "total_points", "minutes", "goals_scored", "assists"
    ]
    available_columns = [col for col in selected_columns if col in selected_team_df.columns]
    selected_team_df[available_columns].to_csv("selected_team.csv", index=False)
    
    # Display the team
    print(selected_team_df[available_columns])
    
    # Save metrics table to CSV
    metrics_table.to_csv("model_metrics.csv", index=False)
    
    # Display metrics table
    print("\n--- Model Metrics ---")
    print(metrics_table.to_string(index=False))
    
    # Export metrics to Excel
    try:
        metrics_table.to_excel("model_metrics.xlsx", index=False)
        print("Metrics exported to Excel successfully.")
    except Exception as e:
        print(f"Could not export to Excel: {e}")
        print("Metrics saved as CSV instead.")