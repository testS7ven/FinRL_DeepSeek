"""
Unified stock trading environment — Risk-First architecture.

State vector:
  [cash, prices×N, shares×N, indicators×N×K, sentiment_hat×N, risk_hat×N]

Three optional modules (all toggleable via flags):
  Module 1 — use_confidence:      use confidence-weighted signals in state
  Module 2 — use_reward_shaping:  add risk penalty to reward
  Module 3 — use_circuit_breaker: block/reduce buys when risk high & sentiment low
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]


class StockTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df,
        initial_cash: float = 1_000_000,
        hmax: int = 100,
        reward_scaling: float = 1e-4,
        # Module flags
        use_confidence: bool = True,
        use_reward_shaping: bool = True,
        use_circuit_breaker: bool = True,
        # Module 2 params
        lambda_risk: float = 0.02,
        # Module 3 params
        cb_risk_threshold: float = 4.0,
        cb_sentiment_threshold: float = 2.0,
    ):
        super().__init__()
        self.df             = df.sort_values(["date", "tic"]).reset_index(drop=True)
        self.dates          = sorted(self.df["date"].unique())
        self.tickers        = sorted(self.df["tic"].unique())
        self.n              = len(self.tickers)
        self.n_indicators   = len(INDICATORS)

        self.initial_cash    = initial_cash
        self.hmax            = hmax
        self.reward_scaling  = reward_scaling

        self.use_confidence      = use_confidence
        self.use_reward_shaping  = use_reward_shaping
        self.use_circuit_breaker = use_circuit_breaker
        self.lambda_risk         = lambda_risk
        self.cb_risk_threshold   = cb_risk_threshold
        self.cb_sentiment_threshold = cb_sentiment_threshold

        # State: cash + prices + shares + indicators + sentiment_hat + risk_hat
        self.state_dim = 1 + self.n + self.n + self.n * self.n_indicators + self.n + self.n

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

        self._build_index()
        self.reset()

    # ── Data indexing ──────────────────────────────────────────────────────────

    def _build_index(self):
        self._idx: dict[any, dict] = {}
        sig_col = "sentiment_hat" if self.use_confidence else "sentiment"
        rsk_col = "risk_hat"      if self.use_confidence else "risk"

        for date in self.dates:
            day = self.df[self.df["date"] == date].set_index("tic")

            def get(col, default):
                return np.array([
                    float(day.loc[t, col]) if t in day.index and col in day.columns else default
                    for t in self.tickers
                ])

            self._idx[date] = {
                "prices":       get("close", 0.0),
                "indicators":   np.stack([get(ind, 0.0) for ind in INDICATORS], axis=1),  # (N, K)
                "sentiments":   get(sig_col, 3.0),
                "risks":        get(rsk_col, 3.0),
                "raw_sentiments": get("sentiment", 3.0),
                "raw_risks":      get("risk", 3.0),
            }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        d = self._idx[self.dates[self.t]]
        return np.concatenate([
            [self.cash],
            d["prices"],
            self.shares,
            d["indicators"].flatten(),
            d["sentiments"],
            d["risks"],
        ]).astype(np.float32)

    def _portfolio_value(self, prices: np.ndarray) -> float:
        return float(self.cash + np.sum(self.shares * prices))

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t      = 0
        self.cash   = float(self.initial_cash)
        self.shares = np.zeros(self.n, dtype=np.float64)
        return self._obs(), {}

    def step(self, actions: np.ndarray):
        d             = self._idx[self.dates[self.t]]
        prices        = d["prices"]
        raw_risks     = d["raw_risks"]
        raw_sentiments= d["raw_sentiments"]
        risk_hat      = d["risks"]

        orders = np.round(actions * self.hmax).astype(int)

        # ── Module 3: Circuit-Breaker ──────────────────────────────────────────
        if self.use_circuit_breaker:
            for i in range(self.n):
                if raw_risks[i] >= self.cb_risk_threshold and raw_sentiments[i] <= self.cb_sentiment_threshold:
                    scale = max(0.0, 1.0 - 0.25 * (raw_risks[i] - 3.0))
                    if orders[i] > 0:
                        orders[i] = 0                       # block buys
                    elif orders[i] < 0:
                        orders[i] = int(orders[i] * scale)  # reduce sells

        prev_value = self._portfolio_value(prices)

        # Execute sells first, then buys
        sell_mask = orders < 0
        buy_mask  = orders > 0

        for i in np.where(sell_mask)[0]:
            if prices[i] <= 0:
                continue
            qty = min(abs(orders[i]), int(self.shares[i]))
            self.shares[i] -= qty
            self.cash += qty * prices[i]

        for i in np.where(buy_mask)[0]:
            if prices[i] <= 0:
                continue
            affordable = int(self.cash // prices[i])
            qty = min(orders[i], affordable)
            self.shares[i] += qty
            self.cash -= qty * prices[i]

        new_value = self._portfolio_value(prices)
        reward = (new_value - prev_value) * self.reward_scaling

        # ── Module 2: Reward Shaping ───────────────────────────────────────────
        if self.use_reward_shaping:
            long_vals    = np.maximum(0.0, self.shares) * prices
            total        = max(new_value, 1.0)
            port_weights = long_vals / total
            weighted_risk = float(np.dot(port_weights, risk_hat))
            penalty = self.lambda_risk * weighted_risk * long_vals.sum() * self.reward_scaling
            reward -= penalty

        # LLM risk factor exposed for CPPO CVaR constraint
        stock_vals   = np.maximum(0.0, self.shares) * prices
        total        = max(new_value, 1.0)
        port_weights = stock_vals / total
        llm_risk_factor = float(np.dot(port_weights, risk_hat)) / 5.0 + 1.0  # ∈ [1.0, 2.0]

        self.t += 1
        done = self.t >= len(self.dates) - 1

        obs  = self._obs() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {"portfolio_value": new_value, "llm_risk_factor": llm_risk_factor}
        return obs, float(reward), done, False, info
