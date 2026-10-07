import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import urllib3
import json
import altair as alt
import xgboost as xgb
import sqlite3


st.set_page_config(page_title="Solar Flare Monitor", layout="wide")

# create the center title
st.title("Solar Flare Monitor", text_alignment='center')
st.markdown("###")



# re-run app every 60 seconds
st.fragment(run_every=60)(lambda:None)()








#creates a function fetch_live_noaa_xrsb
#returns a pandas dataframe
#including the last 6 hours of GOES 18 Long flux data straight from NOAA

def fetch_live_noaa_xrsb():
    



    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"

    http = urllib3.PoolManager()

    try:
        print("Establishing connection with the NOAA GOES Telemtry stream...")
        response = http.request('GET', url, timeout=5.0)

        if response.status == 200:
            raw_data = json.loads(response.data.decode('utf-8'))

            # get raw data from NOAA
            df_stream = pd.DataFrame(raw_data)
          
            df_long = df_stream[df_stream["energy"] == '0.1-0.8nm'].copy().reset_index(drop=True)
            
            df_long['flux'] = df_long['flux'].replace(0.0, np.nan)

            #return most recent data for use in model prediction
            return df_long
    except Exception as e:
        print(f"Failed: {e} ")

            
            

            




#Create a function called get_model_prediction
#Which returns the flux value predicted by
#the XGBOOST regressor model


def get_model_prediction():
# CREATE A NEW TEMPORARY DATABSE FOR THE CURRENT VALUE TO GET A PREDICTION FROM THE MODEL
    df_returned = fetch_live_noaa_xrsb()
    df_current_features = df_returned[['flux']].copy()
    
    

    # CREATE A 240 MINUTE WINDOW LOOKING BACKWARDS
    df_current_features['max_velocity'] = df_current_features['flux'].diff()
    r_velocity = df_current_features['max_velocity'].rolling(window=240, min_periods = 1)
    r_flux = df_current_features['flux'].rolling(window = 240, min_periods = 1)

                
    
    df_current_features.drop(columns =  {'max_velocity'}, inplace=True)

    #start adding features using sliding window -> using log values helps XGBOOST create decision trees
    df_current_features['xrsb_mean'] = np.log10(r_flux.mean() + 1e-12) #constant  of 1 e-12 used to prevent taking log(0) 
    df_current_features['max'] = np.log10(r_flux.max() + 1e-12)
    df_current_features['min'] = np.log10(r_flux.min() + 1e-12)
    df_current_features['stdev'] = (r_flux.std())
    df_current_features['variance'] = (r_flux.var())
    df_current_features['max_velocity'] = np.log10(r_velocity.max().clip(lower=1e-12)) 

    #use current val and last val to find avg slope
    df_current_features['current_val'] = df_current_features['flux']
    df_current_features['last_val'] = df_current_features['flux'].shift(240)

    
    #scale difference using a multiplier to create a logarithmically scaled value
    difference = ((df_current_features['current_val'] - df_current_features['last_val'])/240) * 1e12
    df_current_features['avg_slope'] = np.sign((difference)) * (np.log1p(np.abs(difference)))

    df_current_features.drop(columns={'flux', 'current_val', 'last_val'}, inplace = True)
    


    # GET MODEL PREDICTION
    model = train_model()
    model_prediction = model.predict(df_current_features.iloc[-1:])
    
    return 10 ** model_prediction[0]
    


    
    

#creates a function called train_model which
#returns the regressor trained on datasets found in model_data.db

@st.cache_resource
def train_model():

    # create DB/cursor for raw data
    data_db = sqlite3.connect("model_data.db")
    


    #load model from solar_model.json file and return the model
    regressor = xgb.XGBRegressor()
    regressor.load_model("solar_model.json")
    return regressor



#creates a function called draw_tracker
#draws the current xrsb flux graph
#and displays the model's prediction to the right

def draw_tracker():
    
    df_long = fetch_live_noaa_xrsb()
    # create extra margin to the end of the graph 
    hour_margin = pd.Timedelta(hours=1)
    end_value = pd.to_datetime(df_long['time_tag'].max())
    final_value = end_value + hour_margin
    
    col1, col2 = st.columns([5,3])
    with col1:
        live_chart = alt.Chart(df_long).mark_line(
        #change color and thickness of the line graph
        color = '#ff4b4b',
        strokeWidth = 2.5,
            
        ).encode(
            #add x/y labels to the graph
            x = alt.X('time_tag:T',
                    title='Timeline Horizon (Local Time)',
                    scale = alt.Scale(domain=
                                        [df_long['time_tag'].min(), #set domain from the first value 6 hours ago
                                        final_value])), #to 1 hour in the future

            #scale y logarithmically in order to represent all solar flare events
            y = alt.Y('flux:Q', title = 'Raw Flux Value (W/m²)', axis = alt.Axis(format='.0e'), scale = alt.Scale(type='log', base = 10, domain = [1e-9,1e-2])),



            #create a dynamic tooltip that shows data on hover
            tooltip=[
                alt.Tooltip('time_tag:T', title = 'Time (UTC)', format = '%Y-%m-%d %H:%M'),
                alt.Tooltip('flux:Q', title = 'Raw Flux (W/m²)', format = '.3e')
            ]
            
            
        # add UI changes to graph/text    
        ).properties(
            title='Real-Time Space Weather Tracking Streams',
                
            height=500,
            width='container'
                
        ).configure(
                
            font='Courier New', 
            background='#0e1117' 
        ).configure_axis(
            gridColor='#262730',   
            labelColor='#808495',  
            titleColor='#ffffff', 
            domainColor='#262730'
        ).configure_title(
            color='#ffffff',
            fontSize=14,
            fontStyle='normal',
            anchor = 'middle'
        )


        st.altair_chart(live_chart, width='stretch')
    with col2:
        st.metric(
            label = 'Predicted MAX Horizon',
            value = f"{get_model_prediction():.3e}")
        st.caption('The final forecasted maximum peak for the upcoming 4 hours (GOES 18 Long - XRSB in w/m^2)')
        

        

draw_tracker()