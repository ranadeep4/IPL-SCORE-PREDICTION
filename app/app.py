import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import clean_data, engineer_features
from flask import Flask, render_template, request
import pandas as pd
import joblib
from tensorflow import keras
from collections import defaultdict

app = Flask(__name__)

# Load model results and get best model name
with open(os.path.join(os.path.dirname(__file__), '../saved_models/model_type.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Best model:'):
            model_type = line.split(':', 1)[1].strip()
            break
    else:
        model_type = 'RandomForest'  # fallback

# Load model, encoders, scaler, and feature columns
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../saved_models/model.pkl')
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), '../saved_models/label_encoders.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../saved_models/scaler.pkl')
FEATURE_COLS_PATH = os.path.join(os.path.dirname(__file__), '../saved_models/feature_cols.pkl')

if model_type == 'Keras':
    model = keras.models.load_model(MODEL_PATH)
else:
    model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

# Load player lists by team for dropdowns
# You may want to cache this in production
try:
    ipl_data = pd.read_csv('content/ipl_data.csv')
except:
    ipl_data = pd.read_csv('../content/ipl_data.csv')
team_batsmen = defaultdict(list)
team_bowlers = defaultdict(list)
for _, row in ipl_data.iterrows():
    if row['bat_team'] not in team_batsmen or row['batsman'] not in team_batsmen[row['bat_team']]:
        team_batsmen[row['bat_team']].append(row['batsman'])
    if row['bowl_team'] not in team_bowlers or row['bowler'] not in team_bowlers[row['bowl_team']]:
        team_bowlers[row['bowl_team']].append(row['bowler'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    bat_teams = label_encoders['bat_team'].classes_
    bowl_teams = label_encoders['bowl_team'].classes_
    venues = label_encoders['venue'].classes_
    batsmen = label_encoders['batsman'].classes_
    bowlers = label_encoders['bowler'].classes_
    if request.method == 'POST':
        # Only collect minimal user inputs
        input_dict = {
            'bat_team': request.form['bat_team'],
            'bowl_team': request.form['bowl_team'],
            'venue': request.form['venue'],
            'batsman': request.form['batsman'],
            'bowler': request.form['bowler'],
            'runs': int(request.form['runs']),
            'wickets': int(request.form['wickets']),
            'overs': float(request.form['overs']),
            'striker': int(request.form['striker']),
            'non-striker': int(request.form['non_striker']),
            'runs_last_5': int(request.form['runs_last_5']),
            'wickets_last_5': int(request.form['wickets_last_5']),
            'total': 180  # Placeholder, not used for prediction but needed for feature engineering
        }
        input_df = pd.DataFrame([input_dict])
        input_df = clean_data(input_df)

        # Calculate engineered features
        input_df = engineer_features(input_df)

        # Encode categorical features with fallback for unseen labels
        categorical_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler', 'match_phase']
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                value = input_df[col].values[0]
                if value in le.classes_:
                    input_df[col] = le.transform([value])[0]
                else:
                    input_df[col] = le.transform([le.classes_[0]])[0]

        # Select only the features used in training
        input_features = input_df[feature_cols]
        input_scaled = scaler.transform(input_features)

        if model_type == 'Keras':
            prediction = float(model.predict(input_scaled)[0][0])
        else:
            prediction = float(model.predict(input_scaled)[0])
    return render_template('index.html',
                           bat_teams=bat_teams,
                           bowl_teams=bowl_teams,
                           venues=venues,
                           batsmen=batsmen,
                           bowlers=bowlers,
                           team_batsmen=team_batsmen,
                           team_bowlers=team_bowlers,
                           prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)