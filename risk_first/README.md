# FinRL-DeepSeek — Risk-First Extension

Extension of [FinRL-DeepSeek](https://arxiv.org/abs/2502.07393) (Benhenda, 2025) for the **FinRL Contest 2025, Task 1**.

The original paper combines CPPO (Constrained PPO with CVaR) and DeepSeek-V3 signals for stock trading. This project adds three modules that make the system more defensive during market stress: confidence-weighted signals, a light reward penalty, and a deterministic circuit breaker.

**Course**: AI for Finance — PGE5 2025/2026

---

## What the original paper does

The paper trains a trading agent on 84 NASDAQ stocks (2013–2023) using:

- **PPO** — standard policy gradient, no risk constraint
- **CPPO** — PPO with a CVaR constraint that caps losses in the worst 5% of days
- **DeepSeek-V3** — reads daily news and scores each stock: sentiment 1–5 and risk 1–5

Main finding: in bull markets PPO wins on return; in bear markets CPPO-DeepSeek wins on drawdown.

---

## What this project adds

Three modules, each independent and toggleable via `config.yaml`.

### Module 1 — Confidence-weighted signals

The original code trusts the LLM score at face value. A vague article and a detailed one get the same weight.

Fix: call the LLM three times per article with slightly different temperatures. Measure score variance across the three calls. High variance → low confidence → pull the signal back toward neutral (3).

```text
risk_hat = risk × C + 3 × (1 − C)
```

`C = 1` when all three calls agree. `C = 0` when they disagree completely → signal collapses to neutral regardless of the raw score.

### Module 2 — Reward shaping

The CPPO agent optimises CVaR on trajectory-level returns, but its step-level reward is blind to current LLM risk signals.

Fix: subtract a small penalty proportional to the LLM risk and the size of long positions.

```text
R_t = ΔPortfolio − λ × risk_hat × long_position_value
```

`λ = 0.02` keeps the penalty small enough not to override the CVaR objective. The agent stays active; it just costs slightly more to hold large positions when the LLM is nervous.

### Module 3 — Circuit breaker

A hard filter applied after the network outputs its action, before the order executes.

```python
if raw_risk >= 4 and raw_sentiment <= 2:
    block all buys
    scale existing positions by (1 − 0.25 × (risk − 3))
```

At risk=5: scale=0.5, positions halved. At risk=4: scale=0.75. The raw (unweighted) scores are used here deliberately — the circuit breaker reacts to extremes, not averages.

---

## Project structure

```text
risk_first/
├── config.yaml          # all hyperparameters
├── config_test.yaml     # fast smoke-test (3 epochs, 500 steps)
├── pipeline.py          # train + evaluate in one command
├── run_ablation.py      # runs configurations A through F
├── plot_results.py      # generates publication-ready equity curves
├── kaggle_ablation.ipynb# notebook for fast execution on Kaggle GPUs
│
├── signals/
│   └── llm_signals.py   # Module 1: self-consistency confidence
│
├── data/
│   ├── download.py      # Yahoo Finance + technical indicators
│   └── load_hf.py       # loads benstaf/nasdaq_2013_2023 from HuggingFace
│
├── env/
│   └── trading_env.py   # Gym env with Modules 2 and 3
│
├── training/
│   ├── networks.py      # Actor-Critic MLP
│   ├── buffer.py        # PPOBuffer and CPPOBuffer
│   ├── ppo.py           # PPO trainer
│   └── cppo.py          # CPPO trainer with CVaR + LLM risk factor
│
└── evaluation/
    └── metrics.py       # all 4 contest metrics
```

---

## Setup

```bash
# Linux / macOS
cp risk_first/.env.example risk_first/.env
pip install -r requirements.txt

# Windows
copy risk_first\.env.example risk_first\.env
pip install -r requirements_windows.txt
```

The pre-computed DeepSeek-V3 signals from the original paper are available at `benstaf/nasdaq_2013_2023` on HuggingFace. The pipeline downloads them automatically on first run — no API key needed for data.

---

## Running

```bash
# Smoke test: finishes in ~5 seconds
python -m risk_first.pipeline --config risk_first/config_test.yaml --no-llm

# Full run with HuggingFace data (downloads once, then cached)
python -m risk_first.pipeline

# Full ablation study: trains A through F, saves comparison table
python -m risk_first.run_ablation

# Generate equity curve plots for the paper (requires completed ablation run)
python plot_results.py
```

> **Note on Performance:** Training the 6 configurations (A-F) can take time on CPU. You can use the provided `kaggle_ablation.ipynb` to run the entire pipeline seamlessly on a Kaggle T4x2 GPU instance.

The pipeline stages:

| Step | What happens                    | Output                                           |
| ---- | ------------------------------- | ------------------------------------------------ |
| 1    | Load `benstaf/nasdaq_2013_2023` | `cache/train.csv`, `cache/test.csv`              |
| 2    | Build Gym environment           | in memory                                        |
| 3    | Train PPO or CPPO               | `models/{name}_agent.pth`, `logs/{name}.log`     |
| 4    | Backtest on 2019–2023           | portfolio value series                           |
| 5    | Compute metrics vs QQQ          | `results/{name}_metrics.csv`                     |

Each stage writes to disk. Re-running skips already-completed stages.

---

## Ablation configurations

| Config | Algorithm                   | Confidence | Reward shaping | Circuit breaker |
| ------ | --------------------------- | ---------- | -------------- | --------------- |
| A      | PPO                         | —          | —              | —               |
| B      | PPO + LLM                   | —          | —              | —               |
| C      | CPPO                        | —          | —              | —               |
| D      | CPPO + Confidence           | ✓          | —              | —               |
| E      | CPPO + Confidence + Shaping | ✓          | ✓              | —               |
| F      | Full model                  | ✓          | ✓              | ✓               |

---

## Evaluation metrics

All four FinRL Contest 2025 metrics:

| Metric                   | Definition                                                   |
| ------------------------ | ------------------------------------------------------------ |
| Cumulative Return        | `(final − initial) / initial × 100`                          |
| Sharpe Ratio             | `√252 × mean(daily returns) / std(daily returns)`            |
| Rachev Ratio             | `CVaR_up(5%) / CVaR_down(5%)` — tail upside vs tail downside |
| Outperformance Frequency | `% of days where agent return > QQQ return`                  |

The contest score is the average rank across all four metrics (lower rank = better).

---

## Data

| Dataset             | Source                     | Content                                          |
| ------------------- | -------------------------- | ------------------------------------------------ |
| Prices + indicators | `benstaf/nasdaq_2013_2023` | OHLCV, MACD, RSI, CCI, DX, VIX, turbulence       |
| LLM signals         | Same dataset               | `llm_sentiment` and `llm_risk` from DeepSeek-V3  |
| Benchmark           | Yahoo Finance (QQQ)        | NASDAQ-100 ETF for Outperformance Frequency      |

Train period: 2013–2018. Test period: 2019–2023. 84 NASDAQ tickers.

Signal coverage: ~37% of (date, ticker) pairs have a non-neutral LLM score. The rest default to 3 (neutral).

---

## Original paper

Benhenda, M. (2025). *FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents*. arXiv:2502.07393.

Original repo: [github.com/benstaf/FinRL_DeepSeek](https://github.com/benstaf/FinRL_DeepSeek)
