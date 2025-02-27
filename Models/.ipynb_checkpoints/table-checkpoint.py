import pandas as pd

# Creating the table as a DataFrame
data = {
    "Run": ["Run 1", "Run 2", "Run 3"],
    "goals_scored": [0.161, 0.158, 0.178],
    "minutes": [0.156, 0.103, 0.093],
    "clean_sheets": [0.101, 0.109, 0.130],
    "bps": [0.091, 0.136, 0.115],
    "minutes_influence": [0.081, 0.073, 0.050],
    "bonus": [0.072, 0.080, 0.090],
    "influence": [0.066, 0.071, 0.076],
    "selected_by_percent": [0.064, 0.062, 0.041],
    "assists": [0.060, 0.091, 0.088],
    "goal_assist_interaction": [0.038, 0.046, 0.020],
    "creativity": [0.035, 0.0001, 0.023],
    "ict_index": [0.030, 0.006, 0.042],
    "red_cards": [0.015, 0.009, 0.012],
    "now_cost": [0.011, 0.002, 0.012],
    "goals_conceded": [0.011, 0.006, 0.004],
    "yellow_cards": [0.008, 0.016, 0.004],
    "threat": [0.001, 0.031, 0.022],
    "file": ["feature_importance_20250225-210751.json", "feature_importance_20250226-112508.json", "feature_importance_20250226-141731.json"]
}

df = pd.DataFrame(data)

# Save to CSV
filename = "fpl_feature_importance.csv"
df.to_csv(filename, index=False)

print(f"Table saved as {filename}")
