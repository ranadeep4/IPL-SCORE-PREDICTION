import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from model import (
    build_keras_model,
    build_linear_regression,
    build_random_forest,
    build_gradient_boosting,
    build_xgboost
)
from utils import clean_data, engineer_features, encode_categorical, scale_features
import numpy as np
import os
from tensorflow import keras

data = pd.read_csv('content/ipl_data.csv')
data = clean_data(data)


data = engineer_features(data)

cat_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler', 'match_phase']

data_encoded, label_encoders = encode_categorical(data, cat_cols)

# Define features and target based on correlation analysis from EDA
# Top correlated features: required_run_rate, striker, runs_last_5, wickets_remaining, wickets, non-striker, wickets_last_5, runs, team_batting_strength
feature_cols = [
    # High correlation features (correlation > 0.3)
    'required_run_rate', 'striker', 'runs_last_5', 'wickets_remaining', 'wickets', 'non-striker', 'wickets_last_5',
    
    # Medium correlation features (correlation > 0.1)
    'runs', 'team_batting_strength', 'team_bowling_strength', 'run_rate', 'overs_remaining',
    
    # Categorical features (encoded)
    'bat_team', 'bowl_team', 'venue', 'batsman', 'bowler', 'match_phase',
    
    # Additional context features
    'overs'
]

X = data_encoded[feature_cols]
y = data_encoded['total']

print(f"Training with {len(feature_cols)} features based on EDA insights:")
print(f"Features: {feature_cols}")
print(f"Dataset shape: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Dictionary to store models and their names
models = {
    'LinearRegression': build_linear_regression(),
    'RandomForest': build_random_forest(),
    'GradientBoosting': build_gradient_boosting(),
    'XGBoost': build_xgboost(),
    'Keras': build_keras_model(X_train_scaled.shape[1])
}

results = {}
results_text = []

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Keras':
        # Enhanced training for Keras with early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        model.fit(
            X_train_scaled, y_train, 
            epochs=50, 
            batch_size=32, 
            validation_data=(X_test_scaled, y_test), 
            callbacks=[early_stopping],
            verbose=0
        )
        y_train_pred = model.predict(X_train_scaled).flatten()
        y_test_pred = model.predict(X_test_scaled).flatten()
    else:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred) * 100
    test_r2 = r2_score(y_test, y_test_pred) * 100
    
    results[name] = {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2
    }
    result_line = f"{name}: Train MAE={train_mae:.2f}, Test MAE={test_mae:.2f}, Train R2={train_r2:.2f}%, Test R2={test_r2:.2f}%"
    results_text.append(result_line)
    print(result_line)

# Select best model (lowest test MAE)
best_model_name = min(results, key=lambda k: results[k]['test_mae'])
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

# Save the best model
os.makedirs('saved_models', exist_ok=True)
if best_model_name == 'Keras':
    best_model.save('saved_models/model.pkl')
else:
    joblib.dump(best_model, 'saved_models/model.pkl',protocol=4)

# Save the model type, encoders, and scaler
with open('saved_models/model_type.txt', 'w') as f:
    f.write("Model Results:\n")
    for line in results_text:
        f.write(line + "\n")
    f.write(f"\nBest model: {best_model_name}\n")
joblib.dump(label_encoders, 'saved_models/label_encoders.pkl',protocol=4)
joblib.dump(scaler, 'saved_models/scaler.pkl',protocol=4)

# Save feature columns for prediction
joblib.dump(feature_cols, 'saved_models/feature_cols.pkl',protocol=4)

print(f"\nModel training completed!")
print(f"Best model ({best_model_name}) saved with engineered features.")
print(f"Features used: {len(feature_cols)}")
print(f"Features based on EDA correlation analysis: {feature_cols}") 