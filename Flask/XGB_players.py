import os
import pandas as pd
import numpy as np
import joblib
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load Data
df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Models\fpl_players.csv")

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

# Save the trained model and scaler
model_path = r"C:\Users\LENOVO\Downloads\Models\xgboost_model.pkl"
scaler_path = r"C:\Users\LENOVO\Downloads\Models\scaler.pkl"
joblib.dump(xgb_model, model_path)
joblib.dump(scaler, scaler_path)

# Predictions
df["Predicted Points"] = xgb_model.predict(scaler.transform(df[features]))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, xgb_model.predict(X_test_scaled)))
print(f"Root Mean Squared Error: {rmse:.2f}")

# Define position mapping
position_map = {
    "GK": 1,
    "DEF": 2,
    "MID": 3,
    "FWD": 4
}

# Convert element_type to numeric and ignore missing/unexpected values
df = df[df["element_type"].isin(position_map.keys())]  # Drop rows with unexpected values
df["element_type"] = df["element_type"].map(position_map)

# Select Top 5 Players for Each Position
top_players_dict = {}
for element in position_map.values():  # Ensure all mapped values are processed
    top_players_dict[element] = df[df["element_type"] == element].nlargest(5, "Predicted Points")

# Save Separate Top 5 Lists
top_gk_path = r"C:\Users\LENOVO\Downloads\Models\top_gk.csv"
top_def_path = r"C:\Users\LENOVO\Downloads\Models\top_def.csv"
top_mid_path = r"C:\Users\LENOVO\Downloads\Models\top_mid.csv"
top_fwd_path = r"C:\Users\LENOVO\Downloads\Models\top_fwd.csv"

top_players_dict[1].to_csv(top_gk_path, index=False)
top_players_dict[2].to_csv(top_def_path, index=False)
top_players_dict[3].to_csv(top_mid_path, index=False)
top_players_dict[4].to_csv(top_fwd_path, index=False)

print(f"Top players lists saved at: {top_gk_path}, {top_def_path}, {top_mid_path}, {top_fwd_path}")

# Load Top Players Lists
# Load Top Players Lists
top_gk = pd.read_csv(top_gk_path)
top_def = pd.read_csv(top_def_path)
top_mid = pd.read_csv(top_mid_path)
top_fwd = pd.read_csv(top_fwd_path)

# Generate 5 Unique Teams
teams = []
for i in range(5):
    team = {
        "Goalkeeper": top_gk.iloc[i % len(top_gk)]["second_name"],
        "Defender1": top_def.iloc[i % len(top_def)]["second_name"],
        "Defender2": top_def.iloc[(i+1) % len(top_def)]["second_name"],
        "Defender3": top_def.iloc[(i+2) % len(top_def)]["second_name"],
        "Defender4": top_def.iloc[(i+3) % len(top_def)]["second_name"],
        "Midfielder1": top_mid.iloc[i % len(top_mid)]["second_name"],
        "Midfielder2": top_mid.iloc[(i+1) % len(top_mid)]["second_name"],
        "Midfielder3": top_mid.iloc[(i+2) % len(top_mid)]["second_name"],
        "Forward1": top_fwd.iloc[i % len(top_fwd)]["second_name"],
        "Forward2": top_fwd.iloc[(i+1) % len(top_fwd)]["second_name"],
        "Forward3": top_fwd.iloc[(i+2) % len(top_fwd)]["second_name"]
    }
    teams.append(team)

# Save each team combination as a separate file
for idx, team in enumerate(teams, start=1):
    team_df = pd.DataFrame([team])  # Convert the team dictionary to a DataFrame
    team_path = os.path.join(r"C:\Users\LENOVO\Downloads\Models", f"team_{idx}.csv")  # Define file path
    team_df.to_csv(team_path, index=False)  # Save the team to a CSV file
    print(f"Team {idx} saved at: {team_path}")

