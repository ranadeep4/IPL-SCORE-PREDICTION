import pandas as pd
import joblib
from tensorflow import keras
import os
from utils import clean_data, engineer_features

# Load model results and get best model name
with open('saved_models/model_type.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Best model:'):
            model_type = line.split(':', 1)[1].strip()
            break
    else:
        model_type = 'RandomForest'  # fallback

# Load model, encoders, scaler, and feature columns
if model_type == 'Keras':
    model = keras.models.load_model('saved_models/model.pkl')
else:
    model = joblib.load('saved_models/model.pkl')
label_encoders = joblib.load('saved_models/label_encoders.pkl')
scaler = joblib.load('saved_models/scaler.pkl')
feature_cols = joblib.load('saved_models/feature_cols.pkl')

# Example input (replace with actual input as needed)
input_dict = {
    'bat_team': 'Chennai Super Kings',
    'bowl_team': 'Mumbai Indians',
    'venue': 'Wankhede Stadium',
    'batsman': 'MS Dhoni',
    'bowler': 'Jasprit Bumrah',
    'runs': 100,
    'wickets': 2,
    'overs': 12.0,
    'striker': 5,  # Example index or encoded value
    'non-striker': 3,  # Example index or encoded value
    'runs_last_5': 45,
    'wickets_last_5': 1,
    'total': 180  # Placeholder for feature engineering
}

# Prepare input DataFrame
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
            print(f"Warning: '{value}' not in training data for {col}. Using '{le.classes_[0]}' as fallback.")
            input_df[col] = le.transform([le.classes_[0]])[0]

# Select only the features used in training
input_features = input_df[feature_cols]

# Scale features
input_scaled = scaler.transform(input_features)

# Predict
if model_type == 'Keras':
    prediction = model.predict(input_scaled)[0][0]
else:
    prediction = model.predict(input_scaled)[0]
print(f'Predicted Total Score: {prediction:.2f}') 