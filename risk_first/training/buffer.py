"""Experience replay buffers for PPO and CPPO."""

import numpy as np
import scipy.signal


def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOBuffer:
    """
    On-policy buffer for PPO.
    Stores a fixed number of transitions, then computes GAE advantages.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.97):
        self.obs  = np.zeros((size, obs_dim), dtype=np.float32)
        self.act  = np.zeros((size, act_dim), dtype=np.float32)
        self.rew  = np.zeros(size, dtype=np.float32)
        self.ret  = np.zeros(size, dtype=np.float32)
        self.val  = np.zeros(size, dtype=np.float32)
        self.adv  = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr = self.path_start = 0
        self.max_size = size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs[self.ptr]  = obs
        self.act[self.ptr]  = act
        self.rew[self.ptr]  = rew
        self.val[self.ptr]  = val
        self.logp[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0):
        path  = slice(self.path_start, self.ptr)
        rews  = np.append(self.rew[path], last_val)
        vals  = np.append(self.val[path], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[path] = _discount_cumsum(deltas, self.gamma * self.lam)
        self.ret[path] = _discount_cumsum(rews, self.gamma)[:-1]
        self.path_start = self.ptr

    def get(self) -> dict[str, np.ndarray]:
        assert self.ptr == self.max_size, "Buffer not full"
        self.ptr = self.path_start = 0
        # Normalize advantages
        adv_mean, adv_std = self.adv.mean(), self.adv.std()
        self.adv = (self.adv - adv_mean) / (adv_std + 1e-8)
        return {
            "obs":  self.obs.copy(),
            "act":  self.act.copy(),
            "ret":  self.ret.copy(),
            "adv":  self.adv.copy(),
            "logp": self.logp.copy(),
        }


class CPPOBuffer(PPOBuffer):
    """
    PPO buffer extended with a CVaR penalty term per step.
    The valupdate is subtracted from GAE advantages during finish_path,
    making the agent more conservative on high-risk trajectories.
    """

    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.97):
        super().__init__(obs_dim, act_dim, size, gamma, lam)
        self.valupdate = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, val, logp, valupdate: float = 0.0):
        super().store(obs, act, rew, val, logp)
        self.valupdate[self.ptr - 1] = valupdate

    def finish_path(self, last_val: float = 0.0):
        path   = slice(self.path_start, self.ptr)
        rews   = np.append(self.rew[path], last_val)
        vals   = np.append(self.val[path], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        raw_adv = _discount_cumsum(deltas, self.gamma * self.lam)
        self.adv[path] = raw_adv - self.valupdate[path]  # CVaR penalty applied here
        self.ret[path] = _discount_cumsum(rews, self.gamma)[:-1]
        self.path_start = self.ptr
