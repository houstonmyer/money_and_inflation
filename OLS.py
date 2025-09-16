import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

print("CMD", Path.cwd())

data = pd.read_csv('freddata.csv', parse_dates=['observation_date'], index_col='observation_date')

data["m2_qoq"] = np.log(data["M2SL"]).diff(1) # quarter over quarter M2 growth
data["gdp_qoq"] = np.log(data["GDPC1"]).diff(1)
data["cpi"] = data["CPILFESL_PC1"] / 100 # Convert percentage to decimal
data["m2gdpgrowth"] = data["m2_qoq"] - data["gdp_qoq"] # GDP growth minus M2 growth
data["cpi_shift1"] = data["cpi"].shift(1) # CPI lagged by one period

data = data.dropna() # drops rows with NaN values


def plot_series(data):
    fig, axes = plt.subplots(nrows=len(data.columns), ncols=1, figsize=(10, 8))
    for i, column in enumerate(data.columns):
        data[column].plot(ax=axes[i], title=column)
        axes[i].set_ylabel('Rate of Change (decimal)')
        axes[i].set_xlabel('Date')
    plt.tight_layout()
    plt.show()

plot_series(data[['m2_qoq', 'gdp_qoq', 'cpi', 'm2gdpgrowth']])

def ols_regression(dependent_var, independent_vars):
    X = sm.add_constant(independent_vars)  # Adds a constant term to the predictor
    model = sm.OLS(dependent_var, X).fit()
    print(model.summary())
    
    # Plotting the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(dependent_var.index, dependent_var, label='Actual', color='blue')
    plt.plot(dependent_var.index, model.fittedvalues, label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Date')
    plt.ylabel(dependent_var.name)
    plt.legend()
    plt.show()


ols_regression(data['cpi'], data[['m2gdpgrowth','cpi_shift1']])

pre85 = data.loc[:'1984-12-31'].copy()
post85 = data.loc['1985-01-01':].copy()

print("Pre-1985 OLS Regression Results:")
ols_regression(pre85['cpi'], pre85[['m2gdpgrowth', 'cpi_shift1']])

print("Post-1985 OLS Regression Results:")
ols_regression(post85['cpi'], post85[['m2gdpgrowth', 'cpi_shift1']])\


#with lags

MAX_LAG = 16

for lag in range(1, MAX_LAG + 1):
    data[f'm2gdpgrowth_lag{lag}'] = data['m2gdpgrowth'].shift(lag)

data = data.dropna()

def ols_with_lags (dep, indep, lags):
    lagged_vars = [f'{indep}_lag{lag}' for lag in range(1, lags + 1)]
    X = sm.add_constant(data[lagged_vars])
    model = sm.OLS(dep, X).fit()
    print(model.summary())
    return model

models = {}
for l in range(1, MAX_LAG + 1):
    print(f"OLS Regression with {l} lags:")
    models[l] = ols_with_lags(data['cpi'], 'm2gdpgrowth', l)


for l in range(1, MAX_LAG + 1):
    print(f"lag {l} AIC: {models[l].aic}, BIC: {models[l].bic}, rsquared: {models[l].rsquared}")