from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn as skl
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# create DB/cursor for raw data
data_db = sqlite3.connect("model_data.db")

X_test = pd.read_sql_query('SELECT xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix', data_db)
y_test = pd.read_sql_query('SELECT y_value FROM real_testing_matrix', data_db).squeeze()

X_train = pd.read_sql_query('SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix', data_db)
y_train = pd.read_sql_query('SELECT y_value FROM real_training_matrix', data_db).squeeze()

# creating setup for model
regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6,max_delta_step=3, learning_rate=.01, subsample = .8, colsample_bytree=.8, random_state = 42, n_jobs=-1 )
regressor.fit(X_train, y_train)

# get predictions/actual in log form and watt form
predictions_log = regressor.predict(X_test)
predictions_watts = 10 ** (predictions_log)
y_test_watts = 10 ** (y_test)

# calculate how far off model is on average
log_mae = mean_absolute_error(y_test, predictions_log)
log_mse = mean_squared_error(y_test, predictions_log)

real_mae = mean_absolute_error(y_test_watts, predictions_watts)
real_mse = mean_squared_error(y_test_watts, predictions_watts)

# how far off the model is logistically
print(f" LOG Mean Absolute Error (MAE): {log_mae:.4f}")
print(f" LOG Mean Squared Error (MSE): {log_mse:.4f}")

# how far off the model is in watts
print(f" REAL Mean Absolute Error (MAE): {real_mae:.7f}")
print(f" REAL Mean Squared Error (MSE): {real_mse:.7f}")

# get max prediction vs actual max (in watts)
predicted_max = (np.max(predictions_watts))
actual_max = (np.max(y_test_watts))
print(f"Highest Predictetd Flux: {predicted_max:.4f}")
print(f"Highest actual flux: {actual_max:.4f}")

"""
    Current Evaluation (without hyperoptimaztion) for the complete data set
    LOG MAE: .3157  LOG MSE: .1592 (rounded to 4 decimal places)
    REAL MAE: 4e-7  REAL MSE: 0 (rounded to 7 decimal places)
    Highest Predicted Flux: .0001 vs Highest Actual Flux: .00017

    Notes: Nowcasting model does statistically well
    Next steps: evaluate mae, mse given only higher class flares in order to get better idea of accuracy for those
"""

# create a dataset including only 1e-6 watts/m^2 (logirthmically represented as -6.0) (C class flares) events amd higher
query_features = """
    SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope 
    FROM real_testing_matrix 
    WHERE y_value >= -6.0
"""

query_targets = """
    SELECT y_value
    FROM real_testing_matrix
    WHERE y_value >= -6.0
"""

X_test_high = pd.read_sql_query(query_features, data_db)
# Added .squeeze() to keep the array shapes matched during error evaluation
y_test_high = pd.read_sql_query(query_targets, data_db).squeeze()

# analyze the predictions vs actual values (mae, mse)
predictions_log_high = regressor.predict(X_test_high)
predictions_real_high = 10 ** (predictions_log_high)
y_test_real_high = 10 ** (y_test_high)

# suing the real w/m^2 values find mae, mse
real_mae_high = mean_absolute_error(y_test_real_high, predictions_real_high)
real_mse_high = mean_squared_error(y_test_real_high, predictions_real_high)

# using log values find mae, mse
log_mae_high = mean_absolute_error(y_test_high, predictions_log_high)
log_mse_high = mean_squared_error(y_test_high, predictions_log_high)

# how far off the model is logistically
print(f" LOG HIGH Mean Absolute Error (MAE): {log_mae_high:.4f}")
print(f" LOG HIGH Mean Squared Error (MSE): {log_mse_high:.4f}")

# how far off the model is in watts
print(f" REAL HIGH Mean Absolute Error (MAE): {real_mae_high:.7f}")
print(f" REAL HIGH Mean Squared Error (MSE): {real_mse_high:.7f}")


print(regressor.feature_importances_)
xgb.plot_importance(regressor)
plt.show()

#try seeing whether model can actually predict or is going solely based off curent xrsb


trigger_indices = np.where(y_test_watts >= 1e-5)[0]

# track event timelines to check how the model is tracking
if len(trigger_indices) > 0:
    # Extract index of a massive solar flare event
    index = 0
    while index < len(trigger_indices):
    
        first_flare_row = int(trigger_indices[index])
        
        shifts_backwards = [30,15,5]
        for shift in shifts_backwards:
            # isolate the exact dates/predictions
            flux_15m_before = 10 ** (X_test['xrsb_mean'].iloc[first_flare_row-shift])
            ai_guess_15m_before = predictions_watts[first_flare_row-shift]
            actual_peak_value = y_test_watts[first_flare_row]
            
            # output the expectations vs reality
            print(f"{shift} Mins Before - Ai Guess Flare Peak: {ai_guess_15m_before:.3e} W/m²")
            print(f"{shift} Mins Before - Real Time Input Flux: {flux_15m_before:.3e} W/m²")
            print(f"At Peak        - Actual Flare Peak Value: {actual_peak_value:.3e} W/m²")
            print("*******************")
            index+=239
else:
    print(" No X-class flares were captured in this specific test timeframe.")



# """
#     Notes from testing 3 different x flare events and the ML predictions
#     1. Model does semi well to recognize that a C or lower class will grow (usually around 3 times, rather than the ~10times actual)
#     2. Model typically predicts lower values than the max for M/X to X flares (model doesnt handle massive flares well)
#     3. Model's accuracy typically peaks around 30 min prior to event

#     Updates: will now hyperoptimize patterns to reduce weaknesses and imporve strengths
# """




#fine tune hyper parameters and find which values work best using for loops
lrs = [.01, .05, .1]
depths = [4,8,12]
final_lr = lrs[0]
final_depth = depths[0]

#keep track of success rates of models
real_mses_high_list = []
min_mse_high = 1
min_mae_high = 1


for lr in lrs:
    for depth in depths:
        temporary_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=depth, learning_rate=lr, subsample=.8, colsample_bytree=.8, random_state = 42, n_jobs=-1)
        temporary_regressor.fit(X_train, y_train)

        

        # get predictions/actual for C (and higher) in dataset (gives a more accurate depiction of predicting  massive flares)
        temp_predictions_log = temporary_regressor.predict(X_test_high)
        temp_predictions_watts = 10 ** (temp_predictions_log)

        


    
         


        print("THIS IS THE START OF A NEW COMBO\n")
        print(f"THE LEARNING RATE: {lr}\n")
        print(f"THE MAX DEPTH: {depth}\n")
        print("****************\n")

        """
        using HIGH (c class or higher flares) values 
        find mae, mse
        in log and real (w/m^2) form
        """
        log_mae_high = mean_absolute_error(y_test_high, temp_predictions_log)
        log_mse_high = mean_squared_error(y_test_high, temp_predictions_log)

        real_mae_high = mean_absolute_error(y_test_real_high, temp_predictions_watts)
        real_mse_high = mean_squared_error(y_test_real_high, temp_predictions_watts)
        

        # how far off the model is logistically
        print(f" LOG HIGH Mean Absolute Error (MAE): {log_mae_high:.4f}")
        print(f" LOG HIGH Mean Squared Error (MSE): {log_mse_high:.4f}")

        # how far off the model is in watts
        print(f" REAL HIGH Mean Absolute Error (MAE): {real_mae_high:.7f}")
        print(f" REAL HIGH Mean Squared Error (MSE): {real_mse_high:.7f}")



        real_mses_high_list.append(real_mse_high)
            

  

        if (real_mae_high < min_mae_high):
            min_mae_high = real_mae_high
            final_depth = depth
            final_lr = lr

print("*****************\n")
print(f"BEST MAE comes with a LEARNING RATE OF {final_lr} and a MAX DEPTH of {final_depth} ")
        
        




