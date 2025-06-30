from tensorflow import keras
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

def build_keras_model(input_dim):
    """
    Further enhanced Keras model with more layers and neurons for better accuracy
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='linear')
    ])
    huber_loss = tf.keras.losses.Huber(delta=1.0)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=huber_loss, metrics=['mae'])
    return model

def build_linear_regression():
    return LinearRegression()

def build_random_forest():
    return RandomForestRegressor(n_estimators=100, random_state=42)

def build_gradient_boosting():
    return GradientBoostingRegressor(n_estimators=100, random_state=42)

def build_xgboost():
    return xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0) 