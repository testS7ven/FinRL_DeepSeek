"""
All 4 FinRL Contest metrics + helpers.

Metrics:
  1. Cumulative Return (%)
  2. Sharpe Ratio (annualised)
  3. Rachev Ratio (tail upside / tail downside)
  4. Outperformance Frequency (% of days beating benchmark)  ← was missing
"""

import numpy as np
import pandas as pd
import yfinance as yf


# ── Individual metrics ─────────────────────────────────────────────────────────

def cumulative_return(portfolio_values: np.ndarray) -> float:
    """Total return over the period (%)."""
    return float((portfolio_values[-1] / portfolio_values[0] - 1.0) * 100.0)


def sharpe_ratio(portfolio_values: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio (252 trading days)."""
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    excess        = daily_returns - risk_free_rate / 252.0
    std           = excess.std()
    if std < 1e-10:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)


def rachev_ratio(portfolio_values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Rachev ratio = CVaR_up(α) / CVaR_down(α).
    > 1 means good days are proportionally better than bad days.
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    n       = len(returns)
    k       = max(1, int(n * alpha))
    sorted_r = np.sort(returns)
    cvar_down = float(-sorted_r[:k].mean())
    cvar_up   = float(sorted_r[-k:].mean())
    if cvar_down < 1e-10:
        return 0.0
    return cvar_up / cvar_down


def outperformance_frequency(
    portfolio_values: np.ndarray,
    benchmark_values: np.ndarray,
) -> float:
    """
    % of days where agent daily return > benchmark daily return.
    Both arrays must be value series (not return series).
    """
    agent_ret = np.diff(portfolio_values) / portfolio_values[:-1]
    bench_ret = np.diff(benchmark_values) / benchmark_values[:-1]
    n         = min(len(agent_ret), len(bench_ret))
    if n == 0:
        return 0.0
    outperform = np.sum(agent_ret[:n] > bench_ret[:n])
    return float(outperform / n * 100.0)


def max_drawdown(portfolio_values: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (%)."""
    peak     = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak * 100.0
    return float(drawdown.min())


# ── Benchmark download ─────────────────────────────────────────────────────────

def get_benchmark(ticker: str, start: str, end: str, initial_cash: float = 1_000_000) -> np.ndarray:
    """
    Download a benchmark ETF and normalise to initial_cash.
    Returns a value series aligned to trading days.
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Could not download benchmark {ticker}")
    closes = df["Close"].values.flatten()
    return (closes / closes[0]) * initial_cash


# ── Combined evaluation ────────────────────────────────────────────────────────

def evaluate_all(
    portfolio_values: np.ndarray,
    benchmark_values: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return all 4 contest metrics + max drawdown."""
    return {
        "cumulative_return":        cumulative_return(portfolio_values),
        "sharpe_ratio":             sharpe_ratio(portfolio_values),
        "rachev_ratio":             rachev_ratio(portfolio_values, alpha),
        "outperformance_frequency": outperformance_frequency(portfolio_values, benchmark_values),
        "max_drawdown":             max_drawdown(portfolio_values),
    }


def print_metrics(metrics: dict[str, float], label: str = ""):
    header = f"  {label}" if label else "  Results"
    print(f"\n{'='*50}")
    print(header)
    print(f"{'='*50}")
    for k, v in metrics.items():
        unit = "%" if k in ("cumulative_return", "outperformance_frequency", "max_drawdown") else ""
        print(f"  {k:<30} {v:>10.4f}{unit}")
    print(f"{'='*50}")


def metrics_to_df(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert {model_name: metrics_dict} to a comparison DataFrame."""
    rows = [{"model": name, **m} for name, m in results.items()]
    return pd.DataFrame(rows).set_index("model")
