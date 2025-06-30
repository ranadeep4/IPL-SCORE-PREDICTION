import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def clean_data(df):
    # Drop columns not needed for modeling
    drop_cols = [col for col in ['mid', 'date'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')
    # Drop rows with missing values
    df = df.dropna()
    
    return df

def engineer_features(df):
    """
    Add engineered features based on advanced EDA insights
    """
    df_engineered = df.copy()
    
    # Run rate
    df_engineered['run_rate'] = df_engineered['runs'] / (df_engineered['overs'] + 1e-5)
    
    # Required run rate
    df_engineered['required_run_rate'] = (df_engineered['total'] - df_engineered['runs']) / (20 - df_engineered['overs'] + 1e-5)
    
    # Wickets remaining
    df_engineered['wickets_remaining'] = 10 - df_engineered['wickets']
    
    # Overs remaining
    df_engineered['overs_remaining'] = 20 - df_engineered['overs']
    
    # Match phase
    def get_match_phase(overs):
        if overs <= 6:
            return 'Powerplay'
        elif overs <= 15:
            return 'Middle'
        else:
            return 'Death'
    
    df_engineered['match_phase'] = df_engineered['overs'].apply(get_match_phase)
    
    # Team batting strength
    team_avg_runs = df_engineered.groupby('bat_team')['runs'].mean()
    df_engineered['team_batting_strength'] = df_engineered['bat_team'].map(team_avg_runs)
    
    # Team bowling strength
    team_avg_wickets = df_engineered.groupby('bowl_team')['wickets'].mean()
    df_engineered['team_bowling_strength'] = df_engineered['bowl_team'].map(team_avg_wickets)
    
    # Save engineered data for reference
    engineered_path = os.path.join('content', 'ipl_data_engineered.csv')
    df_engineered.to_csv(engineered_path, index=False)
    
    return df_engineered

# Helper to encode categorical columns
def encode_categorical(data, cat_cols):
    label_encoders = {}
    data_encoded = data.copy()
    for col in cat_cols:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        label_encoders[col] = le
    return data_encoded, label_encoders

# Helper to scale features
def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler 