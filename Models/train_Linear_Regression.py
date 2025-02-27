import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
# Load Dataset
df = pd.read_csv("fpl_players.csv")  # Ensure this contains "total_points"
features = ["goals_scored", "assists", "minutes", "ict_index"]  # Adjust accordingly
target = "total_points"

X = df[features]
y = df[target]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
# Train Linear Regression Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predict Points
y_pred = lr.predict(X_test_scaled)

# Save Predictions
results = pd.DataFrame({"Actual Points": y_test.values, "Predicted Points": y_pred})
results.to_csv("linear_regression_predictions.csv", index=False)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Plot Graph
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Points", color="red")
plt.plot(y_pred, label="Predicted Points", color="blue")
plt.legend()
plt.xlabel("Gameweek")
plt.ylabel("Points")
plt.title("Actual vs. Predicted FPL Points (Linear Regression)")
# Save the figure in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
fig_path = os.path.join(script_dir, "LRO_actual_vs_predicted.png")  # Define full path
plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # Save figure

print(f"Figure saved at: {fig_path}")

plt.show()

plt.show()