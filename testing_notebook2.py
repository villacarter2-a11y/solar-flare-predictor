
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn as skl
from sklearn.metrics import mean_squared_error, mean_absolute_error


# create DB/cursor for raw data
data_db = sqlite3.connect("model_data.db")
data_cursor = data_db.cursor()

#create x/y testing datasets for A class flares
X_test_low = pd.read_sql_query('SELECT xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix WHERE xrsb_mean < -6.0', data_db)
y_test_low = pd.read_sql_query('SELECT y_value FROM real_testing_matrix WHERE xrsb_mean < -6.0', data_db).squeeze()



#create x/y training datssets for A class flares
X_train_low = pd.read_sql_query('SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix WHERE xrsb_mean < -6.0', data_db)
y_train_low = pd.read_sql_query('SELECT y_value FROM real_training_matrix WHERE xrsb_mean < -6.0',data_db).squeeze()


#create x/y testing datasets for C class and lower flares incoming
X_test_high = pd.read_sql_query('SELECT xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix WHERE xrsb_mean >= -6.0', data_db)
y_test_high = pd.read_sql_query('SELECT y_value FROM real_testing_matrix WHERE xrsb_mean >= -6.0', data_db).squeeze()


#create x/y training datasets for C class and higher flares incoming
X_train_high = pd.read_sql_query('SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix WHERE xrsb_mean >= -6.0', data_db)
y_train_high = pd.read_sql_query('SELECT y_value FROM real_training_matrix WHERE xrsb_mean >= -6.0', data_db).squeeze()

#create x/y traing and testing for all values in dataset
X_train_all = pd.read_sql_query('SELECT xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix WHERE xrsb_mean < -6.0', data_db)
y_train_all = pd.read_sql_query('SELECT y_value FROM real_training_matrix WHERE xrsb_mean < -6.0',data_db).squeeze()

X_test_all = pd.read_sql_query('SELECT xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix', data_db)
y_test_all = pd.read_sql_query('SELECT y_value FROM real_testing_matrix', data_db).squeeze()

#make sure correct number of datapoints in data_db

X_train_raw = pd.read_sql_query("SELECT * FROM real_testing_matrix", data_db)
X_train_clean = X_train_raw.drop_duplicates(subset=["minute"], keep="first")

print(f"Original row count in file: {len(X_train_raw)}")
print(f"Cleaned row count for model training: {len(X_train_clean)}")


print(data_cursor.fetchall())

#initialize model settings for both the lower class and higher class models
regressor_low = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, eta=.05, subsample = .8, colsample_bytree=.8, random_state=42 )
regressor_high = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, eta=.05, subsample = .8, colsample_bytree=.8, random_state=42 )
regressor_all = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, eta=.15, subsample = .8, colsample_bytree=.8, random_state=42 )


#fit models to respective datasets  
regressor_high.fit(X_train_high, y_train_high)
regressor_low.fit(X_train_low, y_train_low)
regressor_all.fit(X_train_all, y_train_all)

#convert predicted/actual log values to raw values for both the low and high class regressor models
predicted_low_log = regressor_low.predict(X_test_low)
actual_low_raw = 10 ** y_test_low 
predicted_low_raw =  10 ** predicted_low_log

predicted_high_log = regressor_high.predict(X_test_high)
actual_high_raw = 10 ** y_test_high 
predicted_high_raw = 10 ** predicted_high_log

predicted_all_log = regressor_all.predict(X_test_all)
actual_all_raw = 10 ** y_test_all 
predicted_all_raw = 10 ** predicted_all_log

## track event timelines to check how the model is tracking
# trigger_indices = np.where(actual_all_raw >= 1e-7)[0]
# count=0
# if len(trigger_indices) > 0:
#     # Extract index of a massive solar flare event
#     index = 0
#     while index < len(trigger_indices):
    
#         first_flare_row = int(trigger_indices[index])
        
#         shifts_backwards = [15,5,3,1]
#         for shift in shifts_backwards:
#             # isolate the exact dates/predictions
#             flux_15m_before = 10 ** (X_test_all['xrsb_mean'].iloc[first_flare_row-shift])
#             ai_guess_15m_before = predicted_all_raw[first_flare_row-shift]
#             actual_peak_value = actual_all_raw[first_flare_row]
            
#             # output the expectations vs reality
#             print(f"{shift} Mins Before - Real Time Input Flux: {flux_15m_before:.3e} W/m²")
#             print(f"{shift} Mins Before - Ai Guess Flare Peak: {ai_guess_15m_before:.3e} W/m²")
            
#             print(f"At Peak        - Actual Flare Peak Value: {actual_peak_value:.3e} W/m²")
#             print("*******************")
#             index+=239
#         count+=1
# else:
#     print(" No X-class flares were captured in this specific test timeframe.")

# print(f"Number of rows of data for test used: {240*count}")



# calculate MAE, MSE for low class flare model on low flare data in both LOG form and w/m^2
mae_low_raw = mean_absolute_error(predicted_low_raw, actual_low_raw)
mse_low_raw = mean_squared_error(predicted_low_raw, actual_low_raw)

mae_low_log = mean_absolute_error(predicted_low_log, y_test_low)
mse_low_log = mean_squared_error(predicted_low_log, y_test_low)

# caculate MAE, MSE for high class flare model on high flare data
mae_high_raw = mean_absolute_error(predicted_high_raw, actual_high_raw)
mse_high_raw = mean_squared_error(predicted_high_raw, actual_high_raw)

mae_high_log = mean_absolute_error(predicted_high_log, y_test_high)
mse_high_log = mean_squared_error(predicted_high_log, y_test_high)



# caculate MAE, MSE for all flare model on all data
mae_all_raw = mean_absolute_error(predicted_all_raw, actual_all_raw)
mse_all_raw = mean_squared_error(predicted_all_raw, actual_all_raw)

mae_all_log = mean_absolute_error(predicted_all_log, y_test_all)
mse_all_log = mean_squared_error(predicted_all_log, y_test_all)

# how far off the model is for low values in LOG and w/m^2
print(f" LOG LOW Mean Absolute Error (MAE): {mae_low_log:.4f}")
print(f" LOG LOW Mean Squared Error (MSE): {mse_low_log:.4f}")
print(f" RAW LOW Mean Absolute Error (MAE): {mae_low_raw:.8f}")
print(f" RAW LOW Mean Squared Error (MSE): {mse_low_raw:.8f}")



# how far off the model is for high values in LOG and watts/m^2
print(f" LOG HIGH Mean Absolute Error (MAE): {mae_high_log:.4f}")
print(f" LOG HIGH Mean Squared Error (MSE): {mse_high_log:.4f}")
print(f" RAW HIGH Mean Absolute Error (MAE): {mae_high_raw:.8f}")
print(f" RAW HIGH Mean Squared Error (MSE): {mse_high_raw:.8f}")


# how far off the model is for all values in LOG and watts/m^2
print(f" LOG ALL Mean Absolute Error (MAE): {mae_all_log:.4f}")
print(f" LOG ALL Mean Squared Error (MSE): {mse_all_log:.4f}")
print(f" RAW ALL Mean Absolute Error (MAE): {mae_all_raw:.8f}")
print(f" RAW ALL Mean Squared Error (MSE): {mse_all_raw:.8f}")

# #fine tune hyper parameters and find which values work best using for loops
# lrs = [.1,.15,.2]
# depths = [2,4,6]
# final_lr = lrs[0]
# final_depth = depths[0]

# #keep track of success rates of models
# real_mses_all_list = []
# min_mse_all = 1
# min_mae_all = 1


# for lr in lrs:
#     for depth in depths:
#         temporary_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=depth, learning_rate=lr, subsample=.8, colsample_bytree=.8, random_state = 42, n_jobs=-1)
#         temporary_regressor.fit(X_train_all, y_train_all)

        

#         # get predictions/actual for C (and higher) in dataset (gives a more accurate depiction of predicting  massive flares)
#         temp_predictions_log = temporary_regressor.predict(X_test_all)
#         temp_predictions_watts = 10 ** (temp_predictions_log)

        


    
         


#         print("THIS IS THE START OF A NEW COMBO\n")
#         print(f"THE LEARNING RATE: {lr}\n")
#         print(f"THE MAX DEPTH: {depth}\n")
#         print("****************\n")

#         """
#         using HIGH (c class or higher flares) values 
#         find mae, mse
#         in log and real (w/m^2) form
#         """
#         log_mae_all = mean_absolute_error(y_test_all, temp_predictions_log)
#         log_mse_all = mean_squared_error(y_test_all, temp_predictions_log)

#         real_mae_all = mean_absolute_error(actual_all_raw, temp_predictions_watts)
#         real_mse_all = mean_squared_error(actual_all_raw, temp_predictions_watts)
        

#         # how far off the model is logistically
#         print(f" LOG HIGH Mean Absolute Error (MAE): {log_mae_all:.4f}")
#         print(f" LOG HIGH Mean Squared Error (MSE): {log_mse_all:.4f}")

#         # how far off the model is in watts
#         print(f" REAL HIGH Mean Absolute Error (MAE): {real_mae_all:.7f}")
#         print(f" REAL HIGH Mean Squared Error (MSE): {real_mse_all:.7f}")



#         real_mses_all_list.append(real_mse_all)
            

  

#         if (real_mae_all < min_mae_all):
#             min_mae_all = real_mae_all
#             final_depth = depth
#             final_lr = lr

# print("*****************\n")
# print(f"BEST MAE comes with a LEARNING RATE OF {final_lr} and a MAX DEPTH of {final_depth} ")

# RESULTS WERE LEARNING RATE of .15 and MAX DEPTH of 4


#Create visualization of how well model predicts max xrsb to the actual value

#only look at every hundreth value (ensures model shows accurately for most datapoints)
sample_values= actual_all_raw.iloc[::100]
sample_predicted= predicted_all_raw[::100]



plt.figure(figsize=(12,5))
#create actual value line
plt.plot(sample_values.index, sample_values,
         label = 'Actual Future Peak Maximum',
         color='blue',
         linewidth=1.8,
         alpha=.8
         )

#create predicted value line
plt.plot(sample_values.index, sample_predicted,
         label = 'Predicted Future Peak Maximum',
         color = 'red',
         linestyle="--",
         linewidth=2.0,
         alpha=.9)

#plot with log scale
plt.yscale('log')
plt.title('Predicted vs Actual Future MAX FLUX Monitor')
plt.xlabel("Original databse time line(every 100th row after that)")
plt.ylabel("Max Flux in XRSB (w/m^2)")
plt.grid(True, which='both', linestyle=":", alpha =.4)
plt.legend(fontsize=10, loc='upper left')
plt.show()



# create a visualization of expected vs occuring growth rates

sample_values_xrsb= 10 ** (X_test_all['xrsb_mean'][::100])
sample_predicted_xrsb = predicted_all_raw[::100]
sample_actual_xrsb = actual_all_raw[::100]

actual_growth = sample_actual_xrsb/sample_values_xrsb.values
predicted_growth = sample_predicted_xrsb/sample_values_xrsb.values



plt.figure(figsize=(12,5))
#create actual value line
plt.plot(actual_growth.index, actual_growth,
         label = 'Actual Future Growth',
         color='blue',
         linewidth=1.8,
         alpha=.8
         )

#create predicted value line
plt.plot(actual_growth.index, predicted_growth,
         label = 'Predicted Future Growth',
         color = 'red',
         linestyle="--",
         linewidth=2.0,
         alpha=.9)

#plot with log scale
plt.yscale('log')
plt.title('Predicted vs Actual Future Growth Monitor')
plt.xlabel("Original databse time line(for every 100th)")
plt.ylabel("Max Growth index")
plt.grid(True, which='both', linestyle=":", alpha =.4)
plt.legend(fontsize=10, loc='upper left')
plt.show()





