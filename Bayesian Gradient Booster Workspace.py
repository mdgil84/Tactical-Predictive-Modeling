import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import plotly.graph_objs as go

# Read the Excel dataset
data = pd.read_excel(r'/Users/*********/Desktop/2 FEB 24 SW DATA 2.xlsm', sheet_name='ERC A DAILY')

# Load the future exogenous data
future_exog_data = pd.read_excel(r'/Users/*********/Desktop/2 FEB 24 SW DATA 2.xlsm', sheet_name='Future Exogenus Data')

# Convert 'DATE' to DateTime and set as index for both datasets
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Assuming 'future_exog_data' is loaded with the 'DATE' column containing the forecast period dates
future_exog_data['DATES'] = pd.to_datetime(future_exog_data['DATES'])
future_exog_data.set_index('DATES', inplace=True)
future_exog_data = future_exog_data.asfreq('D')  # Set the frequency to daily

# Ensure the future exogenous data aligns with the forecast period
start_forecast_date = '2023-12-29'
end_forecast_date = '2024-03-04'
forecast_period = pd.date_range(start=start_forecast_date, end=end_forecast_date, freq='D')
future_exog = future_exog_data.reindex(forecast_period)

# Specify the column name containing the time series data
time_series_column = 'ERC A% BY DAY'

# Specify the exogenous columns
exog_columns = ['OVERDUE SERV', 'NMCS ERC-A', 'NMCS ERC-P', 'NMCM ERC-A', 
                'NMCM ERC-P', 'Movement', 'TRAINING', 'CTC', 'COMP SERV', 
                'ALL TRAINING']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[exog_columns], data[time_series_column], test_size=0.2, shuffle=True)

# Define the parameter grid for RandomizedSearchCV
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['auto', 'sqrt', 'log2'], 'bootstrap': [True, False]}

# Create and train the Random Forest Regressor model with a randomized search for hyperparameters    
model = RandomForestRegressor()
grid_search = RandomizedSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Print the best model parameters
print(f"Best model parameters: {best_params}")

# Train the best model on the entire training set
best_model.fit(X_train, y_train)

# Predict the ERC A% for the next 57 days using the best model
# Create and train the random forest regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict the ERC A% for the next 57 days
prediction = model.predict(future_exog[exog_columns])  # Use the same exogenous columns for prediction

# Rest of the code...

# Forecasting with exogenous data
forecast_index = future_exog.index
forecast_data = prediction

# Create DataFrame for Forecasted Data
forecast_df = pd.DataFrame({
    'Forecasted Values': forecast_data
}, index=forecast_index)

# Export Forecast Data to Excel 
excel_output_path = '/Users/*********/Desktop/2FEB_RF_forecasted_data_ND.xlsx'
forecast_df.to_excel(excel_output_path)

# Create an interactive plot with plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[time_series_column], mode='lines', name='Original'))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast_data, mode='lines', name='Forecast'))
fig.update_layout(title='Daily Time Series Forecast', xaxis_title='Date', yaxis_title=time_series_column, hovermode='x')
fig.show()

# Print the location of the Excel file
print(f"Forecast data saved to: {excel_output_path}")

# Print the feature importances
# print(model.feature_importances_)

# Print the model score
print(model.score(X_train, y_train))

# Print the R-squared
print(f"R-squared: {model.score(X_train, y_train)}")

# Print the model parameters
print(f"Model parameters: {model.get_params()}")

# Print F-statistic and p-value for the test data and the model's predictions for the test data
print(f"Test data F-statistic: {model.score(X_test, y_test)}")
print(f"Test data p-value: {model.score(X_test, y_test)}")
print(f"Predicted test data F-statistic: {model.score(X_test, model.predict(X_test))}")
print(f"Predicted test data p-value: {model.score(X_test, model.predict(X_test))}")
