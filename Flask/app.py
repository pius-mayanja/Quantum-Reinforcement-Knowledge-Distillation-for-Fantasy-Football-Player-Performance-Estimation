from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model, scaler, and saved data
model_path = r"C:\Users\LENOVO\Downloads\Models\xgboost_model.pkl"
scaler_path = r"C:\Users\LENOVO\Downloads\Models\scaler.pkl"
players_data_path = r"C:\Users\LENOVO\Downloads\Models\predicted_players.csv"
teams_folder = r"C:\Users\LENOVO\Downloads\Models\teams"  # Folder containing team files

xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(players_data_path)

# Define Features
features = ["goals_scored", "assists", "minutes", "ict_index"]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])  # Allow both GET and POST
def predict():
    if request.method == 'POST':
        first_name = request.form['first_name']
        second_name = request.form['second_name']
        
        # Find player in the dataset
        player_data = df[(df['first_name'] == first_name) & (df['second_name'] == second_name)]
        
        if player_data.empty:
            return render_template("predict.html", prediction_text="Player not found.")
        
        # Predict points
        player_features = player_data[features].values
        player_features_scaled = scaler.transform(player_features)
        predicted_points = xgb_model.predict(player_features_scaled)[0]

        return render_template("predict.html", prediction_text=f"Predicted Total Points: {predicted_points:.2f}")
    
    # Handle GET request (display the form)
    return render_template("predict.html")

@app.route('/teams')
def show_teams():
    # Get all team files in the folder
    team_files = [f for f in os.listdir(teams_folder) if f.startswith("team_") and f.endswith(".csv")]
    team_files.sort()  # Sort files to ensure consistent order

    # Load each team and store in a list
    teams = []
    for team_file in team_files:
        team_path = os.path.join(teams_folder, team_file)
        team_df = pd.read_csv(team_path)
        teams.append(team_df.to_dict(orient='records')[0])  # Convert DataFrame to dictionary

    # Render the teams template with the teams data
    return render_template("teams.html", teams=teams)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

