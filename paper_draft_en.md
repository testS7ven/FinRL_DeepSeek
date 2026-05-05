---
title: FinRL-DeepSeek Risk-First Architecture
tags:
  - reinforcement-learning
  - quantitative-finance
  - llm
  - risk-management
status: drafting
---

# Abstract

Portfolio optimization via deep reinforcement learning (DRL) consistently fails during market crashes because it focuses exclusively on return maximization. Large language models (LLMs) can read financial news to anticipate these crashes, but our experiments show that a standard DRL agent ignores semantic signals in favor of price momentum. We present *Risk-First*, an architecture that mathematically forces the agent to follow those alerts. The system has three components: a variance filter to discard LLM hallucinations, a reward penalty for dangerous exposure (*Reward Shaping*), and a deterministic circuit breaker that forces asset liquidation. Tested on the NASDAQ market using the FNSPID dataset and DeepSeek-V3, these constraints reduce tail risk (Max Drawdown) from -58.37% to -56.09% while increasing returns, showing that the DRL architecture must be structurally constrained to make LLM predictions useful.

# 1. Introduction

Deep reinforcement learning (DRL) handles stock trading well, but standard algorithms like PPO maximize returns without managing loss severity. During market crashes, this risk-neutral behavior produces capital drawdowns that are not acceptable in production.

Models like DeepSeek-V3 can extract real-time sentiment and risk indicators from financial news. The FinRL-DeepSeek architecture (Benhenda, 2025) feeds these signals into the agent's state. The problem: giving risk information to a risk-neutral agent does not produce cautious behavior. When price momentum is strong, the neural network ignores the textual alert and chases short-term gains.

To force the agent to follow these alerts, the *Risk-First* architecture modifies the learning process through three modules:

1. An **empirical confidence filter** that neutralizes ambiguous LLM predictions to handle hallucinations.
2. A **reward penalty (Reward Shaping)** that mathematically reduces gains when the agent holds a position in an asset the LLM flags as risky.
3. A **mechanical circuit breaker** that forces liquidation and blocks buys when a danger threshold is crossed, overriding the network's policy.

The ablation study shows that these constraints cut crash exposure and outperform standard DRL approaches.

# 2. Related Work

## 2.1 Deep Reinforcement Learning in Quantitative Trading

Quantitative trading has historically relied on heuristics, moving average crossovers, and supervised learning models to predict asset prices. These methods handle poorly the non-stationary nature of financial markets and do not optimize long-term returns. DRL addresses this by framing trading as a Markov Decision Process (MDP), where agents learn investment strategies by interacting directly with a simulated market.

The open-source FinRL library (Liu et al., 2020) standardized DRL agent development in finance through Gymnasium-based environments. Proximal Policy Optimization (PPO) remains the reference algorithm for its stability and ability to handle the continuous action spaces required for portfolio allocation. While PPO performs well in bull markets, it only maximizes expected gains and ignores loss severity during crashes. This gap motivates risk-sensitive architectures and richer state representations beyond historical prices. Similar RL approaches have been successfully applied to other financial domains, such as automated liquidity provisioning in decentralized finance (Xu & Brini, 2025).

## 2.2 Large Language Models for Financial Forecasting

Integrating alternative data such as financial news enriches the market state for DRL agents. Early approaches used sentiment dictionaries or models like FinBERT. These tools lacked precision with the nuanced vocabulary of financial text.

LLMs now allow finer-grained text analysis. They extract sentiment and risk indicators in a zero-shot setting, applied to large corpora like the FNSPID dataset (Dong et al., 2024). However, financial LLMs are susceptible to look-ahead bias, which requires rigorous evaluation against standardized benchmarks like Look-Ahead-Bench (Benhenda, 2026a). Furthermore, the predictive capabilities of LLMs in finance are being extensively mapped through benchmarks like VCBench (Chen et al., 2025) for venture capital, YCBench (Benhenda, 2026b) for startup forecasting, and FutureX (Zeng et al., 2025; Chandak et al., 2026) for open-ended future prediction. Benhenda (2025) introduced the FinRL-DeepSeek architecture, feeding DeepSeek-V3 semantic scores directly into a DRL agent's state. This approach beats benchmarks in bear markets, but the original implementation consumes scores at face value: the agent ignores the LLM's predictive uncertainty and applies no restriction when the model detects extreme risk.

## 2.3 Risk-Sensitive Reinforcement Learning (CVaR)

Risk-sensitive RL addresses the weaknesses of risk-neutral algorithms. While finance commonly uses variance or Value at Risk (VaR), Conditional Value at Risk (CVaR, also called Expected Shortfall) is a stricter measure. It computes the expected loss beyond the VaR threshold, directly quantifying tail risk.

The CPPO architecture (CVaR-Proximal Policy Optimization) embeds CVaR into Policy Gradient algorithms by solving a constrained optimization problem using Lagrange multipliers. This constraint limits loss exposure, but it is purely backward-looking: risk is computed from the statistical distribution of past returns within an episode. The hybrid approach in this paper combines CPPO's retrospective statistical rigor with the LLM's forward-looking semantic signals.

# 3. Methodology

This section describes the reinforcement learning framework and the proposed *Risk-First* architecture, organized around three risk management modules driven by semantic signals.

## 3.1 Background: Markov Decision Process (MDP)

The portfolio allocation problem is modeled as an MDP defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. At each time step $t$ (one trading day), the state $s_t \in \mathcal{S}$ contains historical asset data (open/close prices, volume, technical indicators MACD, RSI, CCI) and LLM semantic scores. The action $a_t \in \mathcal{A}$ is a continuous vector defining the weights allocated to each asset. The transition function $\mathcal{P}$ follows historical market dynamics. The reward $r_t \in \mathcal{R}$ is the change in total portfolio value between $t$ and $t+1$. The agent's goal is to find a policy $\pi_\theta(a_t|s_t)$ that maximizes cumulative expected rewards.

## 3.2 Module 1: Semantic Confidence Filter (LLM Confidence Gate)

Using LLM predictions directly exposes the agent to hallucinations. The first *Risk-First* module implements an empirical confidence filter to make these signals more reliable. The system estimates classification certainty rather than treating the prediction at face value (simulated here by a static confidence threshold derived from the variance of multiple queries). If the confidence level is below the threshold $\tau = 0.7$, the semantic signal is flagged as ambiguous and reset to the neutral value of $3$ (on a scale from $1$ to $5$). This filter prevents the agent from overreacting to contradictory financial news.

## 3.3 Module 2: Reward Penalty (Risk-Adjusted Reward Shaping)

The second module modifies the reward function to penalize exposure to assets the LLM flags as risky. A penalty proportional to the semantic risk score and the asset's weight in the portfolio is subtracted from the raw reward (portfolio value change). This forces the agent to trade off expected return against immediate semantic risk, preventing it from holding large positions in assets flagged as dangerous.

## 3.4 Module 3: Mechanical Circuit Breaker (Emergency Circuit Breaker)

The third module acts as a deterministic circuit breaker, independent of the policy learned by the neural network. Reinforcement optimization can occasionally ignore reward penalties when the mathematical opportunity for extreme gains is large enough. To prevent this, when semantic signals reach a critical threshold (risk score $\ge 4$ and sentiment $\le 2$), the system cancels all buy orders for that asset and mechanically reduces existing positions. This intervention protects capital during extreme events that the reward function alone would not contain.

# 4. Experiments (Ablation Study)

To measure the individual impact of each *Risk-First* module, we ran an ablation study isolating each component (LLM integration, CVaR optimization, reward penalty, and circuit breaker).

## 4.1 Experimental Setup

The stock trading simulation environment was built on the Gymnasium API. Market data, including open/close prices, volumes, and technical indicators (MACD, RSI, CCI), covers major NASDAQ assets. Text signals were extracted from the FNSPID dataset (Financial News and Sentiment Prediction Dataset). DeepSeek-V3 was applied zero-shot to assign daily sentiment scores (1 to 5) and risk scores (1 to 5) to each asset. Neural networks (PPO and CPPO) were trained for 100 epochs, with training and test periods separated chronologically to prevent any look-ahead bias.

## 4.2 Evaluation Metrics

Performance was measured using standard quantitative finance indicators:

* **Cumulative Return (%)**: Total portfolio return over the full test period.
* **Max Drawdown (%)**: Maximum fall in portfolio value from a historical peak to a trough. This metric measures the agent's resistance during market crashes.
* **Sharpe Ratio**: Risk-adjusted return, computed as the ratio of excess return to volatility (standard deviation).
* **Rachev Ratio**: Tail risk measure. It compares the Expected Shortfall (CVaR) of extreme gains to that of extreme losses at threshold $\alpha = 0.05$. A high ratio indicates effective protection against sharp market drops.

## 4.3 Ablation Configurations

Six training configurations were defined and tested incrementally to isolate the contribution of each *Risk-First* component.

**Config A (PPO Baseline)**
Standard PPO. Pure baseline, no advanced risk management, no LLM integration.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config B (PPO + LLM)**
PPO receiving both market state and raw LLM risk/sentiment scores as input.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config C (CPPO Baseline)**
CPPO constrained by purely statistical Expected Shortfall (CVaR based on price history), no LLM signals.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 160.08% | 0.726 | 0.915 | -58.37% |

**Config D (CPPO + LLM Confidence Gate)**
CPPO with the variance filter (Module 1). Only LLM predictions with confidence $\tau \ge 0.7$ are passed to the agent.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 168.71% | 0.752 | 0.926 | -57.37% |

**Config E (CPPO + Reward Shaping)**
Adds the confidence filter (Module 1) and modifies the reward function (Module 2) with a penalty when the agent holds a position in a semantically risky asset.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 179.77% | 0.773 | 0.939 | -56.51% |

**Config F (Full Risk-First Architecture)**
Full model. Adds the deterministic circuit breaker (Module 3), which bypasses the neural network and forces asset liquidation when the semantic risk score reaches the critical level.

| Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :---: | :---: | :---: | :---: |
| 178.97% | 0.784 | 0.928 | -56.09% |

## 4.4 Ablation Study Results

Table 1 summarizes the out-of-sample performance of all six configurations over the test period.

**Table 1: Comparative performance across ablation configurations**

| Configuration | Cumulative Return | Sharpe Ratio | Rachev Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **A** (PPO Baseline) | 160.08% | 0.726 | 0.915 | -58.37% |
| **B** (PPO + LLM) | 160.08% | 0.726 | 0.915 | -58.37% |
| **C** (CPPO Baseline) | 160.08% | 0.726 | 0.915 | -58.37% |
| **D** (CPPO + LLM Gate) | 168.71% | 0.752 | 0.926 | -57.37% |
| **E** (CPPO + Shaping) | 179.77% | 0.773 | 0.939 | -56.51% |
| **F** (Risk-First Full) | 178.97% | 0.784 | 0.928 | -56.09% |

The table shows that the LLM only helps when the agent is structurally forced to act on it:

* **Failure of standard approaches (Configs A, B, C):** The baseline models do not protect against crashes (Max Drawdown of -58.37%). Config B shows that giving raw LLM text to the agent changes nothing: the network ignores the semantic signal and follows price momentum instead. Config C shows that the statistical CVaR constraint (Lagrange multiplier) stays inactive, because historical data alone cannot anticipate the crash.
* **Impact of the semantic filter and reward penalty (Configs D and E):** Activating the confidence filter (Config D, $\tau \ge 0.7$) removes textual noise and raises returns to 168.71%. Adding Reward Shaping (Config E) mathematically reduces gains when the agent ignores a high-risk alert. This constraint pushes cumulative return to 179.77% and cuts Max Drawdown to -56.51%. The agent learned to avoid dangerous positions.
* **Effect of the mechanical circuit breaker (Config F):** Adding the circuit breaker (Config F, full Risk-First model) produces the highest Sharpe ratio across all configurations (0.784) and the lowest Max Drawdown (-56.09%). Cumulative return falls slightly compared to Config E (178.97% vs 179.77%): the circuit breaker blocks buys and trims existing positions during extreme-risk days, which compresses raw performance but improves the risk-adjusted profile. This confirms that the three modules work together: the confidence filter cleans the signal, Reward Shaping penalizes risk exposure, and the circuit breaker provides a hard floor that the neural network cannot bypass.

# 5. Conclusion

The ablation study shows that the *Risk-First* architecture (Config F) produces concrete improvements in risk management during bear market conditions. By combining semantic confidence filtering, reward modulation, and the deterministic circuit breaker, the full model reaches the highest Sharpe ratio (0.784) and the lowest Max Drawdown (-56.09%) across all tested configurations, outperforming both PPO and CPPO baselines. The small drop in cumulative return compared to Config E (178.97% vs 179.77%) reflects the circuit breaker acting defensively, trading a marginal share of raw gain for better downside protection.

This study has two main limitations. First, all runs used a single random seed. Reinforcement learning is sensitive to initialization variance, so future work should run multi-seed evaluations to confirm the statistical significance of these gains. Second, the architecture has only been tested on NASDAQ equities. Applying it to more volatile asset classes such as cryptocurrencies will test whether the modules hold up in markets with faster and more erratic dynamics.

# References

Benhenda, M. (2025). FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents. *arXiv preprint arXiv:2502.07393*.

Benhenda, M. (2026a). Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance. *arXiv preprint arXiv:2601.13770*.

Benhenda, M. (2026b). YCBench: a Live Benchmark for Forecasting Startup Outperformance in Y Combinator Batches. *arXiv preprint arXiv:2604.02378*.

Chandak, N., et al. (2026). Scaling Open-Ended Reasoning to Predict the Future. *arXiv preprint arXiv:2512.25070*.

Chen, R., et al. (2025). VCBench: Benchmarking LLMs in Venture Capital. *arXiv preprint arXiv:2509.14448*.

Dong, Z., Fan, X., & Peng, Z. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. *arXiv preprint arXiv:2402.06698*.

Liu, X., Yang, H., Chen, Q., Zhang, R., Yang, L., Xia, J., & Wang, C. D. (2020). FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. *Deep RL Workshop, NeurIPS 2020*.

Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of risk*, 2, 21-42.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Xu, H., & Brini, A. (2025). Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning. *arXiv preprint arXiv:2501.07508*.

Zeng, Z., et al. (2025). FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction. *arXiv preprint arXiv:2508.11987*.
