import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Download historical data for my selected stocks
# Note: Adjust the tickers as needed
ten_stocks = ['NVO', 'LLY', 'JPM', 'RY', 'ORCL', 'NVDA', 'WMT', 'LULU', 'LMT', 'BA']
monthly_prices = yf.download(ten_stocks, start='2020-08-01', end='2025-08-01', interval='1mo', auto_adjust=False)
monthly_adj_close = monthly_prices['Adj Close']

# Extract the monthly adjusted close prices for each stock and convert to a dictionary for easy access
monthly_data = {
    ticker: monthly_adj_close[ticker].values
    for ticker in ten_stocks
}

# Function to calculate monthly growth rate for each ticker
def get_monthly_growth_rate(ticker):
    """
    Calculate the monthly growth rate for a ticker symbol.

    Returns:
    Array of monthly growth rates.
    """
    temp_monthly_data = monthly_data[ticker]
    monthly_growth_rate = (temp_monthly_data[1:] - temp_monthly_data[:-1]) / temp_monthly_data[:-1]
    return monthly_growth_rate

# Calculate monthly growth rates, expected monthly growth rates,
# annualized growth rates, monthly variance, and annualized variance for each stock
monthly_growth_rates = {
    ticker: get_monthly_growth_rate(ticker)
    for ticker in ten_stocks
}

expected_monthly_growth = {
    ticker: np.mean(monthly_growth_rates[ticker])
    for ticker in ten_stocks
}

annualized_growth = {
    ticker: (1 + expected_monthly_growth[ticker]) ** 12 - 1
    for ticker in ten_stocks
}

variance_monthly = {
    ticker: np.var(monthly_growth_rates[ticker], ddof=1)
    for ticker in ten_stocks
}

annualized_variance = {
    ticker: variance_monthly[ticker] * 12
    for ticker in ten_stocks
}

# Convert growth rates to a DataFrame for easier covariance calculation
growth_rates_df = pd.DataFrame(monthly_growth_rates)
cov_matrix_monthly = growth_rates_df.cov()
cov_matrix_annual = cov_matrix_monthly * 12

# Convert expected annual returns to a numpy array for optimization
expected_annual_returns = np.array([annualized_growth[t] for t in ten_stocks])

# Number of stocks we are considering
num_assets = len(ten_stocks)

# Portfolio volatility (standard deviation)
def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix @ weights)

# Portfolio variance (objective function for minimization)
def get_portfolio_variance(weights, cov_matrix):
    return weights @ cov_matrix @ weights

# Base constraint: fully invested portfolio
full_investment_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Max weight bounds for each asset (0% to 100% here)
max_weight_per_asset = 1.0
weight_bounds = [(0, max_weight_per_asset) for _ in range(num_assets)]

# Target returns to evaluate on the frontier
target_annual_returns = np.linspace(expected_annual_returns.min(),
                                    expected_annual_returns.max(), 20)

efficient_portfolio_weights = []

for target_return in target_annual_returns:
    # Constraints: fully invested + achieve target return
    constraints = [
        full_investment_constraint,
        {'type': 'eq', 'fun': lambda w, tr=target_return: w @ expected_annual_returns - tr}
    ]
    
    # Starting guess: equal weighting
    initial_weights = np.ones(num_assets) / num_assets
    
    # Optimize portfolio for given target return
    result = minimize(
        get_portfolio_variance,
        initial_weights,
        args=(cov_matrix_annual,),
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
    )
    
    if result.success:
        efficient_portfolio_weights.append(result.x)
    else:
        print(f"Optimization failed for target return = {target_return:.4f}")

# Compute efficient frontier metrics
efficient_returns = [w @ expected_annual_returns for w in efficient_portfolio_weights]
efficient_volatilities = [get_portfolio_volatility(w, cov_matrix_annual)
                                 for w in efficient_portfolio_weights]

# Prepare DataFrame for plotting
weights_df = pd.DataFrame(efficient_portfolio_weights, columns=ten_stocks)
weights_df['Annualized Return'] = efficient_returns
weights_df = weights_df.sort_values('Annualized Return')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Efficient Frontier", fontsize=13)
# --- Scatter Plot (Left) ---
scatter = ax1.scatter(efficient_volatilities, efficient_returns, 
                      c=range(len(efficient_returns)), cmap='viridis', s=100)

# Improved annotation code
for i, (vol, ret) in enumerate(zip(efficient_volatilities, efficient_returns)):
    ax1.annotate(f'{i+1}', 
                (vol, ret),
                xytext=(5, 5), 
                textcoords='offset points',
                ha='center', 
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

ax1.set_xlabel('Volatility (Risk)')
ax1.set_ylabel('Return')
ax1.set_title('Efficient Frontier Scatter Plot')
ax1.grid()

# --- Bar Chart (Right) ---
weights_df.drop(columns='Annualized Return').plot(kind='bar', stacked=True, ax=ax2, width=0.8)
ax2.set_xticklabels([f'{ret:.1%}' for ret in weights_df['Annualized Return']])
ax2.set_xlabel('Annualized Return')
ax2.set_ylabel('Weights')
ax2.set_title('Asset Weights by Portfolio')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()