# %% [markdown]
# # Forecast Net Demand

# %%
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://jason404:Jason404.top@localhost/postgres?options=-csearch_path%3Dsp-df", echo=False)
conn = engine.connect()

# %% [markdown]
# ## Import data from CSV to PostgreSQL
# 
# This step is used for testing purposes.
# 
# Set `IMPORT_DATA` to `False` to skip this step.

# %%
IMPORT_DATA = False

# %%
import pandas as pd
import datetime as dt

if IMPORT_DATA:
    
    # Load and filer data from csv file
    
    rt_dpr = pd.read_csv('./data/RT_DPR.csv')
    rt_dpr = rt_dpr[['Date', 'Period', 'Demand', 'TCL', 'TransmissionLoss']]
    rt_dpr['TransmissionLoss'] = rt_dpr['TransmissionLoss'].fillna(0)
    rt_dpr = rt_dpr[rt_dpr['Date'] > '2023-06-30']
    rt_dpr = rt_dpr.sort_values(by=['Date', 'Period'])
    rt_dpr.reset_index(drop=True, inplace=True)
    
    vc_per = pd.read_csv('./data/VCData_Period.csv')
    
    rt_dpr.to_sql('RealTime_DPR', conn, if_exists='replace', index=False)
    vc_per.to_sql('VCData_Period', conn, if_exists='replace', index=False)

# %% [markdown]
# ## Data from DB

# %%
import datetime as dt

now = dt.datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H:%M")

period = int(now.strftime("%H")) * 2 + int(now.strftime("%M")) // 30 + 1


if period + 1 > 48:
    next_period = 1
    next_date = now + dt.timedelta(days=1)
    next_date = next_date.strftime("%Y-%m-%d")
else:
    next_period = period + 1
    next_date = date

# next_date = '2024-03-25' # A hard-coded value for testing
# next_period = 33 # A hard-coded value for testing
print(f"Now is {date} {time} Period {period}")
print(f"To predict: {next_date} Period {next_period}")

# %%
rt_dpr = pd.read_sql(f"""
                     SELECT "Date", "Period", "Demand", "TCL", "TransmissionLoss" 
                     FROM "RealTime_DPR" 
                     WHERE ("Date" < '{date}' OR ("Date" = '{date}' AND "Period" < {next_period}))
                     ORDER BY "Date" DESC, "Period" DESC  
                     LIMIT 336
                     """, conn)
rt_dpr.sort_values(by=['Date', 'Period'], inplace=True)
rt_dpr.reset_index(drop=True, inplace=True)
rt_dpr.head(2)

# %%
rt_dpr.tail(2)

# %%
vc_per = pd.read_sql('SELECT * FROM "VCData_Period"', conn)
vc_per.head(2)

# %%
import holidays

# Calculate required data fields

sg_holidays = holidays.country_holidays('SG')

rt_dpr['Total Demand'] = rt_dpr['Demand'] + rt_dpr['TCL'] + rt_dpr['TransmissionLoss']
view = rt_dpr[['Date', 'Period', 'Total Demand']].copy()

def find_tcq(row):
    # print(row)
    date_obj = dt.datetime.strptime(row['Date'], '%Y-%m-%d')
    year = date_obj.year
    quarter = (date_obj.month - 1) // 3 + 1
    
    isWeekend = 1 if date_obj.isoweekday() > 5 else 0
    isPublicHoliday = date_obj in sg_holidays
    
    if isWeekend or isPublicHoliday:
        return vc_per[(vc_per['Year'] == year) & (vc_per['Quarter'] == quarter)]['TCQ_Weekend_PH'].values[0] / 1000
    else:
        return vc_per[(vc_per['Year'] == year) & (vc_per['Quarter'] == quarter)]['TCQ_Weekday'].values[0] / 1000

view['TCQ'] = view.apply(lambda row: find_tcq(row), axis=1)
view['Net Demand'] = view['Total Demand'] - view['TCQ']
view.reset_index(drop=True, inplace=True)
# view.head(2)

# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import glob

resDir = './model/'
resources = os.listdir(resDir)
resources.sort(reverse=True)
newest = resources[0]

# Load the most recent scaler file
scaler_files = glob.glob(resDir + newest + '/*.pkl')
if type(scaler_files) == str:
    scaler = joblib.load(scaler_files)
    print("Loaded scaler:", scaler_files)
else:
    scaler = joblib.load(scaler_files[0])
    print("Loaded scaler:", scaler_files[0])

# Perform data preprocessing as before
data = view.copy()
data['Target'] = data['Net Demand']
data['Target'] = scaler.fit_transform(data['Target'].values.reshape(-1,1))

# Create dataset for prediction
def create_dataset(dataset):
    return np.array([dataset])

predict_X = create_dataset(data['Target'].values)

# Reshape input to be [samples, time steps, features]
predict_X = np.reshape(predict_X, (predict_X.shape[0], predict_X.shape[1], 1))
print(f"Predict_X shape: {predict_X.shape}")

# %% [markdown]
# ## Predict using trained model

# %%
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()
# tf.config.set_soft_device_placement(False)
# tf.debugging.set_log_device_placement(False)

print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))

tf.device('/CPU:0')

# %%
import os
import glob
from keras.models import load_model

model_file = glob.glob(resDir + newest + '/*.keras')
if type(model_file) == list:
    model_file = model_file[0]

# Load the selected model
model = load_model(model_file)

# Print the path of the loaded model for verification
print("Loaded model:", model_file)


# Make predictions
predict_result = model.predict(predict_X)

# Invert predictions to original scale
inverted_predictions = scaler.inverse_transform(predict_result)


# %%
# Print or use the predictions as needed
print(f"Predictions: {inverted_predictions[0][0]}")

# %% [markdown]
# ## Save predictions to PostgreSQL

# %%
# Create a DataFrame with the predicted result
prediction_df = pd.DataFrame({
    'Date': [next_date], 
    'Period': [next_period], 
    'Net_Demand': [inverted_predictions[0][0]]
    })

# Check if the table exists
table_exists = engine.dialect.has_table(conn, 'Predictions')

if table_exists:
    print("Table exists")
    # Append the value if the table exists
    prediction_df.to_sql('Predictions', conn, if_exists='append', index=False)
else:
    print("Table doesn't exist")
    # Create the table if it doesn't exist
    prediction_df.to_sql('Predictions', conn, if_exists='replace', index=False)


# %%
conn.commit()
conn.close()

# %%



