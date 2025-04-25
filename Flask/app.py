from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Define paths
model_path = r"xgboost_model.pkl"
scaler_path = r"scaler.pkl"
players_data_path = r"predicted_players.csv"
teams_folder = r"teams"  # Define teams_folder path

# Define Features - must match exactly with training features
features = [
    "goals_scored", "assists", "minutes", "goals_conceded",
    "creativity", "influence", "threat", "bonus", "bps",
    "ict_index", "clean_sheets", "red_cards", "yellow_cards",
    "selected_by_percent", "now_cost", "points_per_game",
    "penalties_saved", "form", "expected_goal_involvements"
]


# Load pre-trained model, scaler, and saved data
xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(players_data_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        first_name = request.form['first_name'].lower()
        second_name = request.form['second_name'].lower()

        # Find player in the dataset
        player_data = df[
            df['first_name'].str.lower().str.contains(first_name, na=False) &
            df['second_name'].str.lower().str.contains(second_name, na=False)
        ]

        if player_data.empty:
            return render_template("predict.html",
                                prediction_text="Player not found. Please check the name.")

        player = player_data.iloc[0]

        # Check if player is available
        if player["chance_of_playing_next_round"] == 0:
            return render_template("predict.html",
                                prediction_text="Player is unavailable for this game week")

        # Calculate predicted points based on form and points per game
        predicted_points = (0.7 * player["points_per_game"] + 0.3 * float(player["form"])) * 38

        player_name = f"{player['first_name']} {player['second_name']}"
        return render_template("predict.html",
                            prediction_text=f"{player_name}'s Predicted Points: {predicted_points:.2f}")

    return render_template("predict.html")

@app.route('/teams')
def show_teams():
    try:
        # Get all team files in the folder
        team_files = [f for f in os.listdir(teams_folder) if f.startswith("team_") and f.endswith(".csv")]

        if not team_files:
            return render_template("teams.html",
                                error="No team data available. Please run the model first.")

        team_files.sort()
        teams = []

        for team_file in team_files:
            team_path = os.path.join(teams_folder, team_file)
            if os.path.exists(team_path):
                team_df = pd.read_csv(team_path)
                teams.append(team_df.to_dict(orient='records')[0])

        return render_template("teams.html", teams=teams)

    except Exception as e:
        return render_template("teams.html",
                            error=f"Error loading teams: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)