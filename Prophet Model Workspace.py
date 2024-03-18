#Prophet Model Workspace

import pandas as pd
from prophet import Prophet

# Step 1: Read the main data from Excel
data = pd.read_excel('/Users/********/Desktop/2 FEB 24 SW DATA.xlsm', sheet_name='ERC A DAILY')

# Step 2: Prepare the main data
data = data.rename(columns={'DATE': 'ds', 'ERC A% BY DAY': 'y'})
data['ds'] = pd.to_datetime(data['ds'])

# Step 3: Read the exogenous data from Excel
future_exog_data = pd.read_excel('/Users/********/Desktop/2 FEB 24 SW DATA.xlsm', sheet_name='Future Exogenus Data')
future_exog_data['DATES'] = pd.to_datetime(future_exog_data['DATES'])

# Step 4: Create and setup the Prophet model
model = Prophet()
regressors = ['OVERDUE SERV', 'NMCS ERC-A', 'NMCS ERC-P', 'NMCM ERC-A', 'NMCM ERC-P', 'Movement', 
              'TRAINING', 'CTC', 'COMP SERV', 'DONSA', 'ALL TRAINING']
for reg in regressors:
    model.add_regressor(reg)

# Step 5: Fit the model
model.fit(data)

# Step 6: Prepare future dataframe for prediction
future = model.make_future_dataframe(periods=len(future_exog_data))
future = pd.merge(future, future_exog_data, left_on='ds', right_on='DATES')

# Step 7: Make predictions
forecast = model.predict(future)

# Step 8: Save the forecast to an Excel file
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel('/Users/*********/Desktop/2FEB_Prophet_forecast.xlsx')

# Optional: Plotting
model.plot(forecast)
model.plot_components(forecast)
