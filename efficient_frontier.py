import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# ======================================================
# 1. Stock Data
# ======================================================

ten_stocks = ['NVO', 'LLY', 'JPM', 'RY', 'ORCL', 'NVDA', 'WMT', 'LULU', 'LMT', 'BA']

# Download monthly prices
monthly_prices = yf.download(
    ten_stocks, start='2020-06-01', end='2025-06-01',
    interval='1mo', auto_adjust=False
)
monthly_adj_close = monthly_prices['Adj Close']

# Store raw price arrays per ticker
monthly_data = {ticker: monthly_adj_close[ticker].values for ticker in ten_stocks}

# --- Manual Growth Rate + Variance Functions ---
def get_monthly_growth_rate(ticker):
    """Compute monthly % growth for a ticker."""
    data = monthly_data[ticker]
    return (data[1:] - data[:-1]) / data[:-1]

def get_variance_monthly(ticker):
    """Compute sample variance of monthly returns manually."""
    growth_rates = get_monthly_growth_rate(ticker)
    mean = sum(growth_rates) / len(growth_rates)
    squared_diffs = sum((growth - mean) ** 2 for growth in growth_rates)
    return squared_diffs / (len(growth_rates) - 1)

def get_annualized_growth(monthly_growth_rates):
    """Compute compounded annualized growth (CAGR)."""
    growths = 1 + np.array(monthly_growth_rates)
    total_growth = np.prod(growths)
    num_years = len(growths) / 12
    return total_growth ** (1 / num_years) - 1

# --- Compute stats per asset ---
monthly_growth_rates = {ticker: get_monthly_growth_rate(ticker) for ticker in ten_stocks}
expected_monthly_growth = {ticker: np.mean(monthly_growth_rates[ticker]) for ticker in ten_stocks}
annualized_growth = {ticker: get_annualized_growth(monthly_growth_rates[ticker]) for ticker in ten_stocks}
variance_monthly = {ticker: get_variance_monthly(ticker) for ticker in ten_stocks}
annualized_variance = {ticker: variance_monthly[ticker] * 12 for ticker in ten_stocks}

# Monthly returns DataFrame for covariance
growth_rates_df = pd.DataFrame(monthly_growth_rates, index=monthly_adj_close.index[1:])
cov_matrix_monthly = growth_rates_df.cov()
cov_matrix_annual = cov_matrix_monthly * 12


# ======================================================
# 2. Factor Data (Fama-French 3 Factors)
# ======================================================

# Load CSV
factors_df = pd.read_csv("F-F_Research_Data_Factors.csv", skiprows=3)

# Keep only rows with YYYYMM
factors_df.iloc[:, 0] = factors_df.iloc[:, 0].astype(str)
factors_df = factors_df[factors_df.iloc[:, 0].str.match(r"^\d{6}$")]

# Rename + convert
factors_df = factors_df.rename(columns={factors_df.columns[0]: "Date", "Mkt-RF": "MKT_excess"})
factors_df[["MKT_excess", "SMB", "HML", "RF"]] = (
    factors_df[["MKT_excess", "SMB", "HML", "RF"]].astype(float) / 100
)
factors_df["Date"] = pd.to_datetime(factors_df["Date"], format="%Y%m")
factors_df = factors_df.set_index("Date")

# Align to stock data
common_idx = growth_rates_df.index.intersection(factors_df.index)
growth_rates_df = growth_rates_df.loc[common_idx]
factors_aligned = factors_df.loc[common_idx, ["MKT_excess", "SMB", "HML"]]


# ======================================================
# 3. Factor Model Covariance
# ======================================================

n_assets, k = len(ten_stocks), 3
factors = factors_aligned.values
B = np.zeros((n_assets, k))
resid_vars = np.zeros(n_assets)

# Regress each stock on factors
for i, ticker in enumerate(ten_stocks):
    y = growth_rates_df[ticker].values
    X = sm.add_constant(factors)
    model = sm.OLS(y, X).fit()
    B[i, :] = model.params[1:]          # betas (skip intercept)
    resid_vars[i] = np.var(model.resid, ddof=1)

# Factor + idiosyncratic covariance
Sigma_f = np.cov(factors.T, ddof=1) * 12
Sigma_epsilon = np.diag(resid_vars * 12)
cov_matrix_factor = B @ Sigma_f @ B.T + Sigma_epsilon


# ======================================================
# 4. Portfolio Optimization
# ======================================================

# Risk-free asset
risk_free_rate = 0.00
expected_annual_returns = np.array([annualized_growth[t] for t in ten_stocks])
expected_annual_returns_with_cash = np.append(expected_annual_returns, risk_free_rate)

# Add cash row/col
cov_matrix_with_cash = np.zeros((n_assets + 1, n_assets + 1))
cov_matrix_with_cash[:n_assets, :n_assets] = cov_matrix_factor

# Constraints
full_investment_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
weight_bounds_with_cash = [(0, 1) for _ in range(n_assets)] + [(0, 1)]

def get_portfolio_variance(weights, cov_matrix):
    return weights @ cov_matrix @ weights

# Efficient frontier
target_annual_returns = np.linspace(risk_free_rate, expected_annual_returns.max(), 20)
efficient_portfolio_weights = []

for tr in target_annual_returns:
    constraints = [
        full_investment_constraint,
        {'type': 'eq', 'fun': lambda w, tr=tr: w @ expected_annual_returns_with_cash - tr}
    ]
    initial_weights = np.append(np.ones(n_assets) / n_assets, 0)
    result = minimize(get_portfolio_variance, initial_weights,
                      args=(cov_matrix_with_cash,), method='SLSQP',
                      bounds=weight_bounds_with_cash, constraints=constraints)
    if result.success:
        efficient_portfolio_weights.append(result.x)

# Returns & volatilities
efficient_returns = [w @ expected_annual_returns_with_cash for w in efficient_portfolio_weights]
efficient_volatilities = [np.sqrt(w @ cov_matrix_with_cash @ w) for w in efficient_portfolio_weights]

# Sharpe ratios
sharpe_ratio = np.divide(np.array(efficient_returns) - risk_free_rate,
                         efficient_volatilities,
                         out=np.full_like(efficient_returns, np.nan),
                         where=np.array(efficient_volatilities) > 0)
id_best_sharpe = int(np.nanargmax(sharpe_ratio))

# Identify cash portfolio
frontier_weights = np.asarray(efficient_portfolio_weights)
id_cash_portfolio = int(np.where(np.isclose(frontier_weights[:, -1], 1.0, atol=1e-8))[0][0])
mask = np.arange(len(efficient_returns)) != id_cash_portfolio


# ======================================================
# 5. Plotting
# ======================================================

finite = np.isfinite(sharpe_ratio)
quantile_edges = np.linspace(0, 100, 8)
sharpe_bins = np.unique(np.nanpercentile(sharpe_ratio, quantile_edges))
sharpe_cmap = plt.get_cmap('turbo')
sharpe_norm = BoundaryNorm(sharpe_bins, ncolors=sharpe_cmap.N, clip=True)

weights_df = pd.DataFrame(efficient_portfolio_weights, columns=ten_stocks + ['Cash'])
weights_df['Annualized Return'] = efficient_returns
weights_df = weights_df.sort_values('Annualized Return')

fig, (ax_frontier, ax_portfolio_weights) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Efficient Frontier", fontsize=13)

# --- Scatter Plot (Left) ---
scatter = ax_frontier.scatter(
    np.asarray(efficient_volatilities)[mask],
    np.asarray(efficient_returns)[mask],
    c=sharpe_ratio[mask],
    norm=sharpe_norm,
    s=110, edgecolor='k', linewidth=0.5,
)
ax_frontier.scatter(
    efficient_volatilities[id_cash_portfolio],
    efficient_returns[id_cash_portfolio],
    s=110, color='gray', edgecolor='k', linewidth=0.5, zorder=4,
)
ax_frontier.scatter(
    efficient_volatilities[id_best_sharpe],
    efficient_returns[id_best_sharpe],
    marker='*', s=450, facecolors='none', edgecolors='k', linewidth=1.2, zorder=5
)
cbar = plt.colorbar(scatter, ax=ax_frontier, pad=0.01, boundaries=sharpe_bins)
cbar.set_label('Sharpe Ratio')
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.5f}"))

for i, (vol, ret) in enumerate(zip(efficient_volatilities, efficient_returns)):
    ax_frontier.annotate(f'{i+1}', (vol, ret),
        xytext=(5, 5), textcoords='offset points',
        ha='center', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

ax_frontier.set_xlabel('Volatility (Risk)')
ax_frontier.set_ylabel('Return')
ax_frontier.set_title('Efficient Frontier Scatter Plot')
ax_frontier.grid()

# --- Bar Chart (Right) ---
weights_df.drop(columns='Annualized Return').plot(
    kind='bar', stacked=True, ax=ax_portfolio_weights, width=0.8
)
ax_portfolio_weights.set_xticklabels([f'{ret:.1%}' for ret in weights_df['Annualized Return']])
ax_portfolio_weights.set_xlabel('Annualized Return')
ax_portfolio_weights.set_ylabel('Weights')
ax_portfolio_weights.set_title('Asset Weights by Portfolio')
ax_portfolio_weights.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_portfolio_weights.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
