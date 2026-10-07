import xgboost as xgb
import sqlite3
import pandas as pd

# Connect to local db to train model
data_db = sqlite3.connect("model_data.db")
X_train_all = pd.read_sql_query('SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix', data_db)
y_train_all = pd.read_sql_query('SELECT y_value FROM real_training_matrix', data_db)

# Train the regressor
regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, eta=.15, subsample=.8, colsample_bytree=.8, random_state=42)
regressor.fit(X_train_all, y_train_all['y_value'])

# SAVE IT OUT (This file will be tiny!)
regressor.save_model("solar_model.json")
print("Model saved successfully as solar_model.json!")