import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load Data
df = pd.read_csv("fpl_players.csv") 

# Define Features & Target
features = ["goals_scored", "assists", "minutes", "ict_index"]  # Adjust based on your dataset
target = "total_points"

X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions
df["Predicted Points"] = xgb_model.predict(scaler.transform(df[features]))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test_scaled)))
print(f"Root Mean Squared Error: {rmse:.2f}")

# Select Players to Compare
players_to_compare = ["Haaland", "Saka", "Salah"]  # Adjust names
player_data = df[df["second_name"].isin(players_to_compare)]

# Plot Actual vs. Predicted Points for Selected Players
plt.figure(figsize=(8, 5))

x_labels = player_data["second_name"].unique()
x = np.arange(len(x_labels))  # x positions

plt.bar(x - 0.2, player_data.groupby("second_name")["total_points"].mean(), width=0.4, label="Actual Points", color="red")
plt.bar(x + 0.2, player_data.groupby("second_name")["Predicted Points"].mean(), width=0.4, label="Predicted Points", color="blue")

plt.xticks(x, x_labels, rotation=45)
plt.xlabel("Players")
plt.ylabel("Total Points")
plt.title("Actual vs. Predicted Total Points (XGBoost)")
plt.legend()

# Save the figure in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
fig_path = os.path.join(script_dir, "XGB_actual_vs_predicted.png")  # Define full path
plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # Save figure

print(f"Figure saved at: {fig_path}")

plt.show()
