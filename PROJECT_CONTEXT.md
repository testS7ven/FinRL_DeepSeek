# Project Context — FinRL-DeepSeek Risk-First Extension

This document gives a complete briefing on the project: what the challenge is, what has been built, how the code is structured, and what remains to do. It is intended for any collaborator (human or AI) picking up the work.

---

## 1. The challenge

**FinRL Contest 2025, Task 1** — build a stock trading agent that beats the NASDAQ-100 benchmark (QQQ ETF) across four metrics:

| Metric | Definition |
| ------ | ---------- |
| Cumulative Return | `(final − initial) / initial × 100` |
| Sharpe Ratio | `sqrt(252) * mean(daily_returns) / std(daily_returns)` |
| Rachev Ratio | `CVaR_up(5%) / CVaR_down(5%)` — tail upside vs tail downside |
| Outperformance Frequency | `% of days where agent return > QQQ return` |

**Contest score = average rank across the four metrics** (lower rank = better). The test period is 2019–2023 (84 NASDAQ tickers). Train period is 2013–2018.

---

## 2. Starting point — the original paper

**Paper**: Benhenda, M. (2025). *FinRL-DeepSeek: LLM-Infused Risk-Sensitive RL for Trading Agents*. arXiv:2502.07393.

**Repo integrated into**: [github.com/AI4Finance-Foundation/FinRL_DeepSeek](https://github.com/AI4Finance-Foundation/FinRL_DeepSeek)

The paper trains four agents on 84 NASDAQ stocks:

| Agent | Key feature | Result |
| ----- | ----------- | ------ |
| PPO | Standard policy gradient | Best in bull market (2019–2021) |
| CPPO | PPO + CVaR constraint | Safer but lower return |
| PPO-DeepSeek | PPO + LLM sentiment | Marginal improvement |
| CPPO-DeepSeek | CPPO + LLM sentiment + risk | Best in bear market (2022) |

**CPPO** uses a CVaR constraint: it penalises trajectories that fall in the worst `alpha%` of returns. The constraint is implemented via a Lagrange multiplier (`cvarlam`) that is updated each epoch.

**DeepSeek-V3** reads daily news and scores each (date, ticker) pair: `sentiment ∈ [1,5]` and `risk ∈ [1,5]`. ~37% of pairs have a non-neutral signal; the rest default to 3 (neutral).

The paper's original codebase is in `data/`, `envs/`, `training/`, `evaluation/` and requires a Linux server with MPI.

---

## 3. This project — the Risk-First extension

The project is in `risk_first/`. It is a **clean reimplementation** (one file per module, no MPI dependency, runs on Windows/Mac) that adds **three new modules** on top of CPPO-DeepSeek.

### Module 1 — Confidence-weighted signals (`risk_first/signals/llm_signals.py`)

**Problem**: the original code trusts every LLM score at face value. A vague article and a detailed one get the same weight.

**Fix**: call the LLM three times per article at temperatures `[0.0, 0.2, 0.4]`. Measure score variance across the three calls.

```
C = max(0, 1 - std / 2)          # confidence ∈ [0, 1]
risk_hat = risk × C + 3 × (1 - C) # pulled toward neutral when uncertain
```

`C = 1` → three calls agreed → use score at face value.  
`C = 0` → three calls disagreed completely → signal collapses to neutral (3).

**In practice (no live LLM)**: the pre-computed HuggingFace dataset does not include self-consistency data, so `confidence_risk = 0.7` for rows that have a signal, `0.0` for neutral rows. `risk_hat` is thus still computed but with a fixed conservative confidence.

### Module 2 — Reward shaping (`risk_first/env/trading_env.py`)

**Problem**: CPPO optimises CVaR on trajectory-level returns. Its step-level reward is blind to current LLM risk signals.

**Fix**: subtract a small penalty proportional to LLM risk and long position size.

```
R_t = ΔPortfolio × reward_scaling
    − lambda_risk × weighted_portfolio_risk × long_value × reward_scaling
```

`lambda_risk = 0.02` (configured in `config.yaml`). The agent remains active; it just costs slightly more to hold large positions when the LLM signals high risk.

**Important**: reward shaping is disabled during backtesting (`use_reward_shaping=False` in the test env) because it is a training signal, not a real trading cost.

### Module 3 — Circuit breaker (`risk_first/env/trading_env.py`)

**Problem**: even with the CVaR constraint and reward shaping, the agent can still take large positions during extreme events.

**Fix**: a hard filter applied **after** the network outputs its action, **before** the order executes. Uses raw (non-weighted) scores.

```python
if raw_risk >= 4 and raw_sentiment <= 2:
    orders[i] = 0                           # block all buys
    scale = max(0, 1 - 0.25 * (risk - 3))  # scale existing long positions
    # risk=4 → scale=0.75 (lose 25%)
    # risk=5 → scale=0.50 (lose 50%)
```

---

## 4. Architecture

```
FinRL_DeepSeek/
├── README.md
├── requirements.txt
├── risk_first/
│   ├── config.yaml            ← all hyperparameters
│   ├── config_test.yaml       ← fast smoke-test (3 epochs, 500 steps)
│   ├── pipeline.py            ← main entry point (train + evaluate)
│   ├── run_ablation.py        ← runs configs A→F, saves comparison table
│   ├── signals/
│   │   └── llm_signals.py     ← Module 1: self-consistency confidence
│   ├── data/
│   │   ├── download.py        ← Yahoo Finance + stockstats indicators
│   │   └── load_hf.py         ← downloads benstaf/nasdaq_2013_2023 from HuggingFace
│   ├── env/
│   │   └── trading_env.py     ← Gym env — Modules 2 and 3 live here
│   ├── training/
│   │   ├── networks.py        ← ActorCritic MLP (tanh output, learnable log_std)
│   │   ├── buffer.py          ← PPOBuffer + CPPOBuffer (GAE via scipy.signal.lfilter)
│   │   ├── ppo.py             ← PPO trainer
│   │   └── cppo.py            ← CPPO trainer with CVaR + LLM risk factor
│   └── evaluation/
│       └── metrics.py         ← all 4 contest metrics + max_drawdown
└── data/, envs/, training/, evaluation/   ← original baseline (Linux/MPI)
```

### State space

For N tickers (N=84 in production, N=30 in download.py default):

```
obs = [cash(1), prices(N), shares(N), indicators(N×4), sentiment_hat(N), risk_hat(N)]
obs_dim = 1 + N + N + N×4 + N + N = 1 + 8N
```

For N=84: `obs_dim = 673`. For N=30: `obs_dim = 241`.

Indicators: MACD, RSI(30), CCI(30), DX(30).

### Action space

`actions ∈ (-1, 1)^N` — continuous, bounded by tanh output.  
Executed as: `orders = round(actions × hmax)` where `hmax=100`.

### Networks (`training/networks.py`)

- `pi_net`: MLP → tanh → bounded actions
- `v_net`: MLP → scalar value
- `log_std`: learnable parameter (not dependent on state)
- `act()`: decorated with `@torch.no_grad()`

### Buffer (`training/buffer.py`)

- **PPOBuffer**: stores `(obs, act, rew, val, logp)`, computes GAE via `scipy.signal.lfilter`
- **CPPOBuffer**: extends PPOBuffer with `valupdate` field. In `finish_path()`, subtracts `valupdate` from advantages to implement the CVaR penalty

### CPPO CVaR constraint (`training/cppo.py`)

The LLM risk factor from the environment info amplifies the CVaR penalty:

```python
llm_risk_factor = dot(portfolio_weights, risk_hat) / 5.0 + 1.0  # ∈ [1.0, 2.0]
adjusted_D = llm_risk_factor × (ep_ret + val - nu)

if adjusted_D < nu and cvarlam > 0:
    valupdate = delay × cvarlam / (1 - alpha) × (nu - adjusted_D)
    valupdate = min(valupdate, abs(val) × cvar_clip)
```

`nu` is the CVaR return threshold (updated each epoch from the worst `1-alpha` trajectories).  
`cvarlam` is the Lagrange multiplier (increased when `cvar > beta`, decreased otherwise).

---

## 5. Data

**No local preprocessing needed.** The pipeline downloads pre-computed data from HuggingFace on first run.

| Dataset | Source | Contents |
| ------- | ------ | -------- |
| Prices + indicators + LLM signals | `benstaf/nasdaq_2013_2023` on HuggingFace | OHLCV, MACD, RSI, CCI, DX, VIX, turbulence, `llm_sentiment`, `llm_risk` |
| Benchmark | Yahoo Finance `QQQ` | NASDAQ-100 ETF daily closes |

**Signal columns in HuggingFace dataset**:
- `llm_sentiment` → renamed to `sentiment` (1–5, neutral=3)
- `llm_risk` → renamed to `risk` (1–5, neutral=3)
- `sentiment_hat` and `risk_hat`: confidence-weighted versions (computed in `load_hf.py`)
- `confidence_risk = 0.7` when signal ≠ 3 (signal present), `0.0` when neutral

**Download flow** (`load_hf.py`):

1. `hf_hub_download()` pulls `train.parquet` and `test.parquet` from `benstaf/nasdaq_2013_2023`
2. Maps column names, computes `sentiment_hat` / `risk_hat`
3. Saves to `risk_first/cache/train.csv` and `risk_first/cache/test.csv`
4. Subsequent runs skip download (cache exists)

---

## 6. Key hyperparameters

From `risk_first/config.yaml`:

```yaml
training:
  epochs:          100
  steps_per_epoch: 20000
  gamma:           0.995
  lam:             0.97
  clip_ratio:      0.7
  target_kl:       0.35
  pi_lr:           3.0e-5
  vf_lr:           1.0e-4
  train_pi_iters:  100
  train_v_iters:   80
  hidden_sizes:    [512, 512]

cppo:
  alpha:           0.85    # CVaR at 85th percentile
  beta:            3000.0  # max acceptable CVaR loss
  cvar_clip_ratio: 0.8
  delay:           1.0

env:
  lambda_risk:            0.02   # Module 2 penalty coefficient
  cb_risk_threshold:      4.0    # Module 3 activation (raw risk)
  cb_sentiment_threshold: 2.0    # Module 3 activation (raw sentiment)
```

---

## 7. Running the project

```bash
# Smoke test — finishes in ~5 seconds
python -m risk_first.pipeline --config risk_first/config_test.yaml --no-llm

# Full CPPO run with LLM signals (HuggingFace data, cached)
python -m risk_first.pipeline

# PPO only
python -m risk_first.pipeline --algo ppo

# Full ablation study (configs A → F, saves ablation_summary.csv)
python -m risk_first.run_ablation

# Skip already-done configs
python -m risk_first.run_ablation --skip A B
```

**Training monitor** (in logs):
- `AvgEpRet` — should increase over epochs
- `KL` — should stay below `1.5 × target_kl` (early stopping kicks in)
- `nu` — CVaR threshold, should drift toward better returns
- `cvarlam` — Lagrange multiplier, should stabilise > 0 if constraint is binding

---

## 8. Ablation configurations

| Config | Algorithm | Confidence (M1) | Reward shaping (M2) | Circuit breaker (M3) |
| ------ | --------- | --------------- | ------------------- | -------------------- |
| A | PPO | — | — | — |
| B | PPO + LLM | — | — | — |
| C | CPPO | — | — | — |
| D | CPPO + LLM | ✓ | — | — |
| E | CPPO + LLM | ✓ | ✓ | — |
| F | Full model | ✓ | ✓ | ✓ |

Results from a smoke test (3 epochs, 500 steps, 2020–2022 data, PPO+LLM):

```
cumulative_return       248.49%
sharpe_ratio              0.99
rachev_ratio              0.96
outperformance_frequency 57.20%
max_drawdown            -36.78%
```

Note: these are from a 3-epoch test, not a full 100-epoch run. Full training has not been run yet.

---

## 9. Current state

| Component | Status |
| --------- | ------ |
| `risk_first/` full codebase | Done |
| HuggingFace data loader (`load_hf.py`) | Done |
| All 4 contest metrics including Outperformance Frequency | Done |
| Smoke test passing (~5 seconds) | Done |
| `requirements.txt` | Done (corrected: added `python-dotenv`, `PyYAML`, `websockets>=12.0`) |
| Root `README.md` | Done |
| `risk_first/README.md` | Done |
| `.gitignore` (root + risk_first/) | Done |
| Git repo initialized + pushed | Done — [github.com/testS7ven/FinRL_DeepSeek](https://github.com/testS7ven/FinRL_DeepSeek) |
| Pre-flight code review (May 2026) | Done — see section 10 for the two fixes applied |
| Symmetric KL early-stop fix in ppo.py / cppo.py | Done |
| Equity-curve persistence in pipeline.py | Done |
| Full 100-epoch training run | **In progress** — Configs A to E are done, F is currently computing on Kaggle GPU. |
| Ablation study A→F | **In progress** — Awaiting F. |
| Multi-seed aggregation in run_ablation.py | **Skipped** — Sticking to a single seed due to the May 5th deadline constraints. |
| Paper (Markdown draft, NeurIPS format) | **Done** — `paper_draft.md` is fully structured and written, awaiting Config F numbers. |
| Visualisation scripts | **Done** — `plot_results.py` created to generate publication-ready equity curves. |
| Kaggle GPU integration | **Done** — `kaggle_ablation.ipynb` created and actively used for training. |

---

## 10. Known issues and fixes applied

| Issue | Fix |
| ----- | --- |
| `ModuleNotFoundError: websockets.sync` | Added `websockets>=12.0` to requirements |
| PyTorch warning on KL scalar conversion | Added `.detach()` before `.mean()` in ppo.py and cppo.py |
| `CastError` when loading HuggingFace dataset | Use `hf_hub_download()` + `pd.read_csv()` instead of `datasets` library loader |
| `Outperformance Frequency` missing from evaluation | Implemented in `metrics.py` |
| KL early-stop never triggered when `kl < 0` (the approximate KL `mean(logp_old - logp_new)` can be negative due to sampling noise — the smoke logs showed KL = -6 to -20 across all PPO iterations, meaning the policy never short-circuited for the full 100 iters) | Changed test to `if abs(kl) > 1.5 * target_kl` in `ppo.py` and `cppo.py` |
| Equity curves not persisted after each run (only the 4 final metrics ended up in `*_metrics.csv`, making paper-quality figures impossible) | Added `np.save(f"{run_name}_equity.npy", portfolio_arr)` and `f"{run_name}_benchmark.npy"` in `pipeline.run_pipeline` |
| Smoke test (`config_test.yaml`) uses `steps_per_epoch=500` but a full episode on 2013–2018 is ~1500 trading days, so no episode terminates within an epoch and `AvgEpRet` stays at 0 across all 3 epochs (looks broken but isn't — it's just that `ep_rets` is never populated) | None applied. Recommendation: bump `steps_per_epoch` to 2000 in `config_test.yaml` for any future smoke. The full `config.yaml` already uses 20 000, which gives ~13 episodes per epoch. |

---

## 11. Dependencies

```
numpy>=1.24,<2.0    pandas>=2.0         scipy>=1.10
torch>=2.0          gymnasium>=0.28
yfinance>=0.2       websockets>=12.0    stockstats>=0.6
openai>=1.0
datasets>=2.14      huggingface_hub>=0.17
PyYAML>=6.0         python-dotenv>=1.0
matplotlib>=3.7
mpi4py>=3.1         (baseline only, Linux)
```

---

## 12. What to do next

### Pre-flight checks before launching the ablation

These have already been done in the most recent code review:

1. ✓ Verified pipeline + run_ablation wire A→F correctly (`env_overrides` per config, test env forces `use_reward_shaping=False`).
2. ✓ Verified M1/M2/M3 are toggled by config flags through `pipeline.build_env`.
3. ✓ Verified the HF cache (`risk_first/cache/{train,test}.csv`, 84 tickers, all signal columns present).
4. ✓ Applied the symmetric KL early-stop fix (see section 10).
5. ✓ Added equity-curve persistence (`*_equity.npy` + `*_benchmark.npy` per config).

Still **TODO** before launching:

6. Cleanup smoke-test artefacts so the new ablation outputs aren't mixed with old ones:
   ```
   rm -f risk_first/models/{ppo_baseline,ppo_llm,test_full}_agent.pth
   rm -f risk_first/results/{ppo_baseline,ppo_llm}_metrics.csv
   rm -f risk_first/results/ablation_results.json
   rm -f risk_first/logs/{ppo_baseline,ppo_llm}.log
   ```
   This is purely cosmetic — the new `run_ablation.py` saves under names `A_PPO_baseline_*`, `B_PPO_LLM_*`, etc., so there's no collision with the leftover files. But it's cleaner to start from an empty `models/` and `results/`.

7. Re-run the smoke with `steps_per_epoch ≥ 2000` to confirm `AvgEpRet ≠ 0` and the KL early-stop fires correctly:
   ```
   python -m risk_first.pipeline --config risk_first/config_test.yaml --no-llm
   ```

8. **Decide single-seed vs multi-seed.** Currently `run_ablation.py` runs each config once (`seed=42` in `config.yaml`). For a paper-quality table you want 3 seeds and report mean ± std. This requires ~30 lines of changes to `run_ablation.py` (a `for seed in [42, 123, 7]:` loop + aggregation). Trade-off: 1 seed ≈ 12–35 h total runtime, 3 seeds ≈ 3 days–4 days.

### Runtime estimate (refined from smoke logs)

Smoke test measured: PPO without LLM ≈ 3.4 ms/step, PPO with LLM ≈ 10 ms/step (84 tickers, single CPU thread). The PPO update cost scales with `steps_per_epoch` × `train_pi_iters`.

| Config family | Per epoch | × 100 epochs | 6 configs total |
| ------------- | --------- | ------------ | --------------- |
| CPU, no LLM (A, C) | ~2 min | ~3 h | included below |
| CPU, with LLM (B, D, E, F) | ~4–5 min | ~7–8 h | — |
| **Total CPU, 1 seed** | — | — | **~30–35 h** (≈ 1.5 day) |
| **Total GPU, 1 seed** | — | — | **~10–15 h** (overnight) |
| **Total CPU, 3 seeds** | — | — | **~4 days** |

Calibrate empirically: launch config A first and look at the `Time` field in the log after 5 epochs. If `Time ≈ 120s`, then config A ≈ 3.3 h, F ≈ 8–10 h, full ablation ≈ 25–35 h on CPU.

### Optional optimisations (not required for first run)

- **Vectorise the circuit-breaker** (`trading_env.py` lines 130–137): the `for i in range(self.n)` loop in Python is the dominant cost for config F. Numpy mask version ≈ 5 lines, ~2–3× speedup expected.
- **GPU support**: `cppo_train` and `ppo_train` don't call `.to(device)` on the network. With CUDA available, adding 4 lines (move agent + tensors to `cuda`) would reduce the PPO-update cost from ~30s to ~3s per epoch on bigger configs.

### Then

9. **Wait for Config F** — The `kaggle_ablation.ipynb` notebook is currently finishing the computation for Config F.

10. **Finalise the paper** — Plug Config F results into `paper_draft.md`, run `plot_results.py` to generate the final equity curve plots, and convert the Markdown text to LaTeX using the official NeurIPS Overleaf template.

11. **Review loop** — As per submission requirements, submit the LaTeX source to ChatGPT, Claude, Grok, and Gemini for a NeurIPS-level review. Iterate if any of them rejects the paper.

12. **Final Submission** — Submit to OpenReview, cross-post to HAL/arXiv, and ensure the GitHub repository is public and updated.
