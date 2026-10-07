from sunpy.net import Fido, attrs as a
import sunpy.timeseries as ts
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb

#create DB/cursor for raw data
model = sqlite3.connect("model_data.db")














# create 30 day start dates for looping
start_dates = pd.date_range(start="2013-08-01", end = "2016-08-01", freq = "MS")


# loop through all 5 years of data used for training and add to SQL dataset
for start in start_dates:

    # set end time to start + 30 days
    end = start + pd.offsets.MonthEnd(1)

    #make datetomes recognizeable by SunPy Fido
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print (f"Batch of times: {start_str} to {end_str}")

    #try adding values into dataframe
    try:


    



        results = Fido.search(a.Time(start_str, end_str),
                            a.Instrument('XRS'),
                            a.Resolution('avg1m'))

        files = Fido.fetch(results)
        goes_data = ts.TimeSeries(files, concatenate = True)
        flux_df = goes_data.to_dataframe()
        flux_df = flux_df.reset_index()
        



        # Eliminate all rows with unreliable data (first handle xrsb quality, then negative flux values)
        flux_df = flux_df[flux_df['xrsb_quality'] == 0]
        flux_df['xrsb'] = flux_df['xrsb'].clip(lower=0)


        # downsample to average flux for each minute
        flux_df = flux_df.rename(columns={'index': 'minute'})


      

        print(flux_df.head(241))


        #calculate 1 minute velocity of flux value
        flux_df['flux_velocity'] = flux_df['xrsb'].diff()

        r_velocity = flux_df['flux_velocity'].rolling(window=240)

        #create a 240 min sliding window 
        r_xrsb = flux_df['xrsb'].rolling(window = 240)

        features_df = pd.DataFrame(index = flux_df.index)
        features_df['minute'] = pd.to_datetime(flux_df['minute']).dt.strftime('%Y-%m-%d %H:%M:%s')



        #start adding features using sliding window -> using log values helps XGBOOST create decision trees
        features_df['xrsb_mean'] = np.log10(r_xrsb.mean() + 1e-12) #constant used to prevent taking log(0) 
        features_df['min'] = np.log10(r_xrsb.min() + 1e-12)
        features_df['max'] = np.log10(r_xrsb.max() + 1e-12)
        features_df['stdev'] = (r_xrsb.std())
        features_df['variance'] = (r_xrsb.var())
        features_df['max_velocity'] = (r_velocity.max().clip(lower=1e-12)) #again constant to prevent log(0)

        #use current val and last val to find avg slope
        features_df['current_val'] = flux_df['xrsb']
        features_df['last_val'] = flux_df['xrsb'].shift(240)

        #scale difference using a multiplier to create a logarithmically scaled value
        difference = ((features_df['current_val'] - features_df['last_val'])/240) * 1e12
        features_df['avg_slope'] = np.sign((difference)) * (np.log1p(np.abs(difference)))

        # create column of y values for XGBOOST to learn
        features_df['y_value'] = np.log10(flux_df['xrsb'].shift(-240).rolling(window=240).max() + 1e-12)

        #get rid of unnecessary columns and first 239 rows that dont have the prev data
        features_df = features_df.drop(columns = ['current_val', 'last_val'])
        features_df = features_df.dropna()

        #add model data into SQL database
        model_data = features_df
        model_data.to_sql("real_training_matrix", model, if_exists='append', index = False)


        # reset dataframes
        flux_df = pd.DataFrame()
        features_df = pd.DataFrame()
    except Exception as e:
            print(f"Error at start time: {start_str}. Reason: {e}")
















