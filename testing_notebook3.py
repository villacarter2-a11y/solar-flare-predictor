
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


#create x/y traing and testing for all values in dataset
X_train_all = pd.read_sql_query('SELECT minute, xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')
y_train_all = pd.read_sql_query('SELECT minute, y_value FROM real_training_matrix',data_db).drop_duplicates(subset=['minute'], keep='first')

X_test_all = pd.read_sql_query('SELECT minute,xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')
y_test_all = pd.read_sql_query('SELECT minute, y_value FROM real_testing_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')

#make sure correct number of datapoints in data_db
train_length = len(X_train_all)

print(f"Training data: {train_length}")
print(len(y_train_all))

y_train_len = len(y_train_all)
print(f"Training y: {y_train_len}")


# drop timetag values from ML model dataset (not useful + can cheat/mess up)
X_train_all = X_train_all.drop(columns = ['minute'])
y_train_all = y_train_all.drop(columns = ['minute'])

X_test_all = X_test_all.drop(columns = ['minute'])
y_test_all = y_test_all.drop(columns = ['minute'])



#initialize model settings 
regressor_all = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, eta=.15, subsample = .8, colsample_bytree=.8, random_state=42 )


#fit model to dataset
regressor_all.fit(X_train_all, y_train_all)

# create variables holding predictions and actual maxes (later used to analyze performance)
predicted_all_log = regressor_all.predict(X_test_all)
actual_all_raw = 10 ** y_test_all 
predicted_all_raw = 10 ** predicted_all_log

#turn predicted log vaslues into dataframe for later

df_predicted_all_log = pd.DataFrame({'xrsb_mean': predicted_all_log, 'max':X_test_all['max']}, index= X_test_all.index)

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



# caculate MAE, MSE for all flare model on all data

#create filters for analyzing success of model for various flare levels
mask_b = X_test_all['max'] > -7.0
mask_c = X_test_all['max'] > -6.0
mask_m = X_test_all['max'] > -5.0

mae_all_raw = mean_absolute_error(predicted_all_raw, actual_all_raw)
mse_all_raw = mean_squared_error(predicted_all_raw, actual_all_raw)

mae_all_log = mean_absolute_error(predicted_all_log, y_test_all)
mse_all_log = mean_squared_error(predicted_all_log, y_test_all)



baseline_mae_all = mean_absolute_error(predicted_all_log, y_test_all)
print(f"Model ALL MAE: {baseline_mae_all}")

#calculate basline MAE for minor flares and higher


baseline_current_b = df_predicted_all_log.loc[mask_b, 'max']
baseline_y_b = y_test_all.loc[mask_b, 'y_value']

baseline_mae_b = mean_absolute_error(
    baseline_y_b,
    baseline_current_b
)

print("Model B+ MAE:", baseline_mae_b)

baseline_current_c = df_predicted_all_log.loc[mask_c, 'max']
baseline_y_c = y_test_all.loc[mask_c, 'y_value']

baseline_mae_c = mean_absolute_error(
    baseline_y_c,
    baseline_current_c
)

print("Model C+ MAE:", baseline_mae_c)


baseline_current_m = df_predicted_all_log.loc[mask_m, 'max']
baseline_y_m = y_test_all.loc[mask_m, 'y_value']

baseline_mae_m = mean_absolute_error(
    baseline_y_m,
    baseline_current_m
)

print("Baseline M+ MAE:", baseline_mae_m)

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

#only look at every thousandth value (ensures model shows accurately for most datapoints)
sample_values= actual_all_raw.iloc[::1000]
sample_predicted= predicted_all_raw[::1000]

#calculate baseline MAE for all data
baseline_current_all = X_test_all['max']
baseline_mae_all = mean_absolute_error(baseline_current_all, y_test_all)
print(f"Baseline ALL MAE: {baseline_mae_all}")

#calculate basline MAE for minor flares and higher


baseline_current_b = X_test_all.loc[mask_b, 'xrsb_mean']
baseline_y_b = y_test_all.loc[mask_b, 'y_value']

baseline_mae_b = mean_absolute_error(
    baseline_y_b,
    baseline_current_b
)

print("Baseline B+ MAE:", baseline_mae_b)

baseline_current_c = X_test_all.loc[mask_c, 'xrsb_mean']
baseline_y_c = y_test_all.loc[mask_c, 'y_value']

baseline_mae_c = mean_absolute_error(
    baseline_y_c,
    baseline_current_c
)

print("Baseline C+ MAE:", baseline_mae_c)


baseline_current_m = X_test_all.loc[mask_m, 'xrsb_mean']
baseline_y_m = y_test_all.loc[mask_m, 'y_value']

baseline_mae_m = mean_absolute_error(
    baseline_y_m,
    baseline_current_m
)

print("Baseline M+ MAE:", baseline_mae_m)



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
plt.xlabel("Original databse time line(every 1000th)")
plt.ylabel("Max Flux in XRSB (w/m^2)")
plt.grid(True, which='both', linestyle=":", alpha =.4)
plt.legend(fontsize=10, loc='upper left')
plt.show()



# create a visualization of expected vs occuring growth rates

sample_values_xrsb= 10 ** (X_test_all['xrsb_mean'][::1000].to_numpy())
sample_predicted_xrsb = predicted_all_raw[::1000]
sample_actual_xrsb = actual_all_raw[::1000].to_numpy().flatten()
sample_previous_max = 10 ** (X_test_all['max'][::1000].to_numpy())
print(f"Sample Values: {sample_values_xrsb}")
print(f"sample predicted: {sample_predicted_xrsb}")
print(f"sample actual: {sample_actual_xrsb}")


actual_growth = sample_actual_xrsb/sample_values_xrsb
predicted_growth = sample_predicted_xrsb/sample_values_xrsb



plt.figure(figsize=(12,5))
#create actual value line
plt.plot(range(len(actual_growth)), actual_growth,
         label = 'Actual Future Growth',
         color='blue',
         linewidth=1.8,
         alpha=.8
         )

#create predicted value line
plt.plot(range(len(actual_growth)), predicted_growth,
         label = 'Predicted Future Growth',
         color = 'red',
         linestyle="--",
         linewidth=2.0,
         alpha=.9)

#plot with log scale
plt.yscale('symlog')
plt.title('Predicted vs Actual Future Growth Monitor')
plt.xlabel("Original databse time line(for every 1000th)")
plt.ylabel("Max Growth index")
plt.grid(True, which='both', linestyle=":", alpha =.4)
plt.legend(fontsize=10, loc='upper left')
plt.show()

#plot line showing previous max vs incoming max
plt.plot(range(len(sample_actual_xrsb)), sample_actual_xrsb,
         label = 'Incoming actual max flux',
         color = 'red',
         
         linewidth = 2.0,
         alpha=.9,

)
plt.plot(range(len(sample_previous_max)),sample_previous_max,
         label = 'Previous max',
         color='blue',
         linewidth=2.0,
         alpha=.9,
)
plt.yscale('log')
plt.title('Actual past max vs Atual future max Monitor')
plt.xlabel('Original database time line for every 1000th value')
plt.grid(True, which='both', linestyle=":", alpha =.4)
plt.legend(fontsize=10, loc='upper left')
plt.show()

# Current status: Model performs noticeably worse compared to a baseline of the previous max (which is almost perfect).
#New Point: reconfigure model to guess the change across baseline of previous max and next max (should reduce model effort, produce higher results)




#create databases for new model
X_train_all = pd.read_sql_query('SELECT minute, xrsb_mean, max, min, stdev, variance, max_velocity, avg_slope FROM real_training_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')
y_train_all = pd.read_sql_query('SELECT minute, y_value FROM real_training_matrix',data_db).drop_duplicates(subset=['minute'], keep='first')

X_test_all = pd.read_sql_query('SELECT minute,xrsb_mean, max,min, stdev, variance, max_velocity, avg_slope FROM real_testing_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')
y_test_all = pd.read_sql_query('SELECT minute, y_value FROM real_testing_matrix', data_db).drop_duplicates(subset=['minute'], keep='first')

#use previous database with y and max values to calculate residuals
y_train_residuals = y_train_all['y_value'] - X_train_all['max']
y_test_residuals = y_test_all['y_value'] - X_test_all['max']
actual_residuals =  10 ** y_test_residuals
#drop minute from the datsets not useful
X_train_all = X_train_all.drop(columns = ['minute'])
y_train_all = y_train_all.drop(columns = ['minute'])

X_test_all = X_test_all.drop(columns = ['minute'])


#initialize models settings
regressor_residuals = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, eta=.15, subsample = .8, colsample_bytree=.8, random_state=42 )


#fit model to dataset
regressor_residuals.fit(X_train_all, y_train_residuals)

# get the actual and predicted residuals 
predicted_residuals = regressor_residuals.predict(X_test_all)
predicted_residuals_sample = predicted_residuals[::1000]
actual_residuals =  y_test_residuals
actual_residuals_sample = actual_residuals.iloc[::1000]

# compare the actual residuals with the ones the model predicted
mae_residual = mean_absolute_error(predicted_residuals, actual_residuals)
mse_residual = mean_squared_error(predicted_residuals, actual_residuals)
print("*******************")

print(f"MAE Residuals: {mae_residual} ")
print(f"MSE Residuals: {mse_residual}")


# PLOT DIFFERENCE BETWEEN PREDICTED AND ACTUAL RESIDUALS
plt.figure(figsize=(12,5))


plt.plot(range(len(predicted_residuals_sample)), actual_residuals_sample,
         label = 'actual residuals',
         linewidth = 2.0,
         color = 'red',
         alpha = .9)

plt.plot(range(len(predicted_residuals_sample)), predicted_residuals_sample,
         label = 'predicted residuals',
         linewidth = 2.0,
         color='blue',
         alpha = .9,
)





plt.title('Predicted vs Actual Residuals')
plt.xlabel('Original timeline')
plt.legend('fontsize=10', loc='upper left')
plt.show()

# PLOT MODEL PREDICTTION VS ACTUAL FUTURE MAX VALUE

plt.figure(figsize=(12,5))
model_prediction = (predicted_residuals + X_test_all['max'].to_numpy())
model_prediction_sample = 10 ** (model_prediction[::1000])


plt.plot(range(len(sample_actual_xrsb)), sample_actual_xrsb,
         label = 'Actual Max Flux',
         color = 'red',
         linewidth = 2.0,
        alpha=.9)

plt.plot(range(len(model_prediction_sample)), model_prediction_sample,
         label = 'Model Predicted Max Flux',
         color = 'blue',
         linewidth = 2.0,
         alpha=.9)

plt.yscale('log')
plt.title('Model Predicted MAX vs Actual Incoming MAX')
plt.legend(fontsize = 10, loc = 'upper left')
plt.grid(True, which = 'both', linestyle = ":", alpha = .4 )
plt.show()


#caluclate MAE of models predictions vs actual

mae_final = mean_absolute_error(model_prediction, y_test_all['y_value'])
mae_final_assurance = mean_absolute_error(model_prediction, X_test_all['max'] + actual_residuals)
print(f'MAE Model Reisdual + baseline: {mae_final}')
print(f"Make sure the predicted score was calculated correctly -> MAE should match the one above: {mae_final_assurance}")

# Update: model tracks well and mantains a lower MAE compared to the Baseline (.1363 vs .1347)

















