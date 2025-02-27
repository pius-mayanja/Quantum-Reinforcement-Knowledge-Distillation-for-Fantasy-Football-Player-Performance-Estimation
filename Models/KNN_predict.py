import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Data
df = pd.read_csv("fpl_players.csv") # Ensure this dataset contains "total_points"
features = ["goals_scored", "assists", "minutes", "ict_index"]  # Adjust based on your dataset
target = "total_points"

X = df[features]
y = df[target]

# Split into Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict Points
y_pred = knn.predict(X_test_scaled)

# Save Results
results = pd.DataFrame({"Actual Points": y_test.values, "Predicted Points": y_pred})
results.to_csv("knn_predictions.csv", index=False)

# Plot Graph
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Points", color="red")
plt.plot(y_pred, label="Predicted Points", color="blue")
plt.legend()
plt.xlabel("Gameweek")
plt.ylabel("Points")
plt.title("Actual vs. Predicted FPL Points (KNN)")

# Save the figure in the same folder as the script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
fig_path = os.path.join(script_dir, "KNN_actual_vs_predicted.png")  # Define full path
plt.savefig(fig_path, dpi=300, bbox_inches="tight")  # Save figure

print(f"Figure saved at: {fig_path}")

plt.show()
plt.show()
