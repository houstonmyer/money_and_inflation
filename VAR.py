import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

print("CMD", Path.cwd())

data = pd.read_csv('freddata.csv', parse_dates=['observation_date'], index_col='observation_date')

data["m2_qoq"] = np.log(data["M2SL"]).diff(1) # quarter over quarter M2 growth
data["gdp_qoq"] = np.log(data["GDPC1"]).diff(1) # Quarter over quarter GDP growth
data["cpi"] = data["CPILFESL_PC1"] / 100 # Convert percentage to decimal
data["m2gdpgrowth"] = data["m2_qoq"] - data["gdp_qoq"] # GDP growth minus M2 growth

data = data.dropna() # drops rows with NaN values

#visualize the data

def plot_series(data):
    fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, figsize=(10, 8))
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i], title=column)
        axes[i].set_ylabel('Rate of Change (decimal)')
        axes[i].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

plot_series(data[['m2_qoq', 'gdp_qoq', 'cpi', 'm2gdpgrowth']])

post85 = data.loc['1985-01-01':].copy()
pre85 = data.loc[:'1984-12-31'].copy()

def check_stationarity(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] < 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary.")

for column in data.columns:
    print(f"Checking stationarity for {column}:")
    check_stationarity(data[column])
    print("\n")

for column in post85.columns:
    print(f"Checking stationarity for {column} (post-1985):")
    check_stationarity(post85[column])
    print("\n")

for column in pre85.columns:
    print(f"Checking stationarity for {column} (pre-1985):")
    check_stationarity(pre85[column])
    print("\n")

def var_model(data, maxlags=8):
    print(f"Fitting VAR model with maxlags={maxlags}")
    model = VAR(data)
    results = model.fit(maxlags)
    print(results.summary())
    print(results.is_stable())
    #print("Forecasting with VAR model:")
    #lag_order = results.k_ar
   #forecast = results.forecast(data.values[-lag_order:], steps=8)

    #print("visualizing the forecast")
    #forecast_index = pd.date_range(start=data.index[-1], periods=9)
    #forecast_data = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)
    #plot_series(pd.concat([data, forecast_data]))

var_model(data[['m2_qoq', 'gdp_qoq', 'cpi']], maxlags=8)
var_model(post85[['m2_qoq', 'gdp_qoq', 'cpi']], maxlags=8)
var_model(pre85[['m2_qoq', 'gdp_qoq', 'cpi']], maxlags=8)




