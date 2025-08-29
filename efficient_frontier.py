import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm

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

# Calculate monthly growth rates, expected monthly growth rates,
# annualized growth rates, monthly variance, and annualized variance for each stock

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

# Function to calculate monthly variance for each ticker
def get_variance_monthly(ticker):
    """
    Calculate the monthly variance for a ticker symbol.

    Returns:
    Monthly variance.
    """
    growth_rates = get_monthly_growth_rate(ticker)
    sum_of_squared_differences = 0
    for growth in growth_rates:
        difference = growth - (sum(growth_rates) / len(growth_rates))
        squared_difference = difference ** 2
        sum_of_squared_differences += squared_difference
    variance_monthly = sum_of_squared_differences / (len(growth_rates) - 1)
    return variance_monthly

variance_monthly = {
    ticker: get_variance_monthly(ticker)
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

# Including cash as an option
# First, add cash as an "asset" with zero volatility and zero correlation
# We'll represent cash as an additional asset in our calculations
# Add risk-free rate (annual)
risk_free_rate = 0.00  # 0% annual return

# Extend the expected returns array to include cash
expected_annual_returns_with_cash = np.append(expected_annual_returns, risk_free_rate)

# Extend the covariance matrix to include cash (which has zero variance and zero covariance)
cov_matrix_with_cash = np.zeros((num_assets + 1, num_assets + 1))
cov_matrix_with_cash[:num_assets, :num_assets] = cov_matrix_annual

# Number of assets now includes cash
num_assets_with_cash = num_assets + 1

# Max weight bounds for each asset (0% to 100% here)
max_weight_per_asset = 1.0
weight_bounds = [(0, max_weight_per_asset) for _ in range(num_assets)]

# Update weight bounds - cash can be any positive weight (up to 100%)
weight_bounds_with_cash = [(0, max_weight_per_asset) for _ in range(num_assets)] + [(0, 1)]

# Update target returns range to include returns below the minimum stock return
target_annual_returns = np.linspace(risk_free_rate, expected_annual_returns.max(), 20)

efficient_portfolio_weights = []

for target_return in target_annual_returns:
    # Constraints: fully invested + achieve target return
    constraints = [
        full_investment_constraint,
        {'type': 'eq', 'fun': lambda w, tr=target_return: w @ expected_annual_returns_with_cash - tr}
    ]
    
    # Starting guess: equal weighting for risky assets, zero for cash
    initial_weights = np.append(np.ones(num_assets) / num_assets, 0)
    
    # Optimize portfolio for given target return
    result = minimize(
        get_portfolio_variance,
        initial_weights,
        args=(cov_matrix_with_cash,),
        method='SLSQP',
        bounds=weight_bounds_with_cash,
        constraints=constraints,
    )
    
    if result.success:
        efficient_portfolio_weights.append(result.x)
    else:
        print(f"Optimization failed for target return = {target_return:.4f}")

# Compute efficient frontier metrics
efficient_returns = [w @ expected_annual_returns_with_cash for w in efficient_portfolio_weights]
efficient_volatilities = [get_portfolio_volatility(w, cov_matrix_with_cash) for w in efficient_portfolio_weights]

## After computing efficient_returns and efficient_volatilities, THIS IS FOR SHARPE RATIO

# --- Sharpe ratios calculation ---
# Sharpe ratio: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
# Avoid division by zero by using np.divide with where condition
portfolio_returns = np.asarray(efficient_returns)
portfolio_volalities = np.asarray(efficient_volatilities)
sharpe_ratio = np.divide(portfolio_returns - risk_free_rate, portfolio_volalities, 
                out=np.full_like(portfolio_returns, np.nan), where=portfolio_volalities > 0)  # avoid /0
id_best_sharpe = int(np.nanargmax(sharpe_ratio))  # index of max-Sharpe portfolio

# Identify the 100% cash portfolio and build a mask to exclude it from the colormapped scatter
frontier_weights = np.asarray(efficient_portfolio_weights)
id_cash_portfolio = int(np.where(np.isclose(frontier_weights[:, -1], 1.0, atol=1e-8))[0][0])  # last col is 'Cash'
mask = np.arange(len(efficient_returns)) != id_cash_portfolio


finite = np.isfinite(sharpe_ratio)
# Quantile edges → 8 bands (can change if I want more/less)
quantile_edges = np.linspace(0, 100, 8)                   # 0,12.5,...,100
sharpe_bins = np.nanpercentile(sharpe_ratio, quantile_edges)
sharpe_bins = np.unique(sharpe_bins)                       # guard against duplicates
sharpe_cmap = plt.get_cmap('turbo')
sharpe_norm = BoundaryNorm(sharpe_bins, ncolors=sharpe_cmap.N, clip=True)

# Prepare DataFrame for plotting - include cash in the weights
weights_df = pd.DataFrame(efficient_portfolio_weights, columns=ten_stocks + ['Cash'])
weights_df['Annualized Return'] = efficient_returns
weights_df = weights_df.sort_values('Annualized Return')

fig, (ax_frontier, ax_portfolio_weights) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Efficient Frontier", fontsize=13)

# --- Scatter Plot (Left) ---
# draw all non-cash points colored by Sharpe
scatter = ax_frontier.scatter(
    np.asarray(efficient_volatilities)[mask],
    np.asarray(efficient_returns)[mask],
    c=sharpe_ratio[mask],
    norm=sharpe_norm,
    s=110,
    edgecolor='k',
    linewidth=0.5,
)

# draw the 100% cash point in gray (not part of the colormap), no label/number
ax_frontier.scatter(
    efficient_volatilities[id_cash_portfolio],
    efficient_returns[id_cash_portfolio],
    s=110,
    color='gray',
    edgecolor='k',
    linewidth=0.5,
    zorder=4,
)

# draw the best Sharpe point with a star marker. This is the maximum Sharpe ratio portfolio
ax_frontier.scatter(
    efficient_volatilities[id_best_sharpe],
    efficient_returns[id_best_sharpe],
    marker='*', s=450,
    facecolors='none', edgecolors='k', linewidth=1.2,
    zorder=5
)

cbar = plt.colorbar(scatter, ax=ax_frontier, pad=0.01, boundaries=sharpe_bins)
cbar.set_label('Sharpe Ratio')
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))


# Annotation code to number the points on the scatter plot
for i, (vol, ret) in enumerate(zip(efficient_volatilities, efficient_returns)):
    ax_frontier.annotate(f'{i+1}', 
        (vol, ret),
        xytext=(5, 5), 
        textcoords='offset points',
        ha='center', 
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

ax_frontier.set_xlabel('Volatility (Risk)')
ax_frontier.set_ylabel('Return')
ax_frontier.set_title('Efficient Frontier Scatter Plot')
ax_frontier.grid()

# --- Bar Chart (Right) ---
weights_df.drop(columns='Annualized Return').plot(kind='bar', stacked=True, ax=ax_portfolio_weights, width=0.8)
ax_portfolio_weights.set_xticklabels([f'{ret:.1%}' for ret in weights_df['Annualized Return']])
ax_portfolio_weights.set_xlabel('Annualized Return')
ax_portfolio_weights.set_ylabel('Weights')
ax_portfolio_weights.set_title('Asset Weights by Portfolio')
ax_portfolio_weights.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_portfolio_weights.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
