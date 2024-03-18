import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_excel(r'/Users/*******/Desktop/Readiness Data Pool/Stalwart_OverDue_ERCA_ERCP_TNG.xlsm', sheet_name='Sheet1')

# Convert 'DATE' to DateTime and set as index
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Define the target series
orr_series = data['ERC A% BY DAY']

# Define exogenous variables
exog_variables = data[['OVERDUE SERV', 'NMCS ERC-A', 'NMCS ERC-P', 'NMCM ERC-A', 'NMCM ERC-P', 'Movement', 'CTC', 'COMP SERV', 'DONSA']]

# Check for missing and infinite values
print("Missing Values:", exog_variables.isna().sum())
print("Infinite Values:", np.isinf(exog_variables).sum())

# SARIMAX model parameters (you should determine these values)
p, d, q = 1, 1, 1  # Replace with your values
P, D, Q, s = 1, 1, 1, 52  # Replace with your values

# Define and fit the SARIMAX Model
model = SARIMAX(orr_series, exog=exog_variables, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

# Print model summary
print(results.summary())

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(orr_series, label='Original')
plt.plot(results.fittedvalues, color='red', label='Fitted')
plt.legend()
plt.show()
