"""PPO trainer."""

import time
import numpy as np
import torch
from pathlib import Path

from .networks import ActorCritic
from .buffer import PPOBuffer


def ppo_train(
    env,
    cfg: dict,
    save_dir: str = "risk_first/models",
    run_name: str = "ppo",
    log_path: str | None = None,
) -> list[float]:
    """
    Train a PPO agent on `env`.
    Returns the portfolio value history from the final epoch (for evaluation).
    """
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden  = cfg.get("hidden_sizes", [512, 512])

    agent  = ActorCritic(obs_dim, act_dim, hidden)
    pi_opt = torch.optim.Adam(
        list(agent.pi_net.parameters()) + [agent.log_std], lr=cfg["pi_lr"]
    )
    v_opt  = torch.optim.Adam(agent.v_net.parameters(), lr=cfg["vf_lr"])

    buf = PPOBuffer(
        obs_dim, act_dim,
        size=cfg["steps_per_epoch"],
        gamma=cfg["gamma"],
        lam=cfg.get("lam", 0.97),
    )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    last_portfolio_history: list[float] = []

    obs, _   = env.reset()
    ep_ret   = 0.0

    for epoch in range(cfg["epochs"]):
        t0              = time.time()
        ep_rets: list[float]         = []
        portfolio_hist: list[float]  = []

        for step in range(cfg["steps_per_epoch"]):
            act, val, logp = agent.act(obs)
            next_obs, rew, done, _, info = env.step(act)

            buf.store(obs, act, rew, val, logp)
            portfolio_hist.append(info.get("portfolio_value", 0.0))
            ep_ret += rew
            obs     = next_obs

            timeout = (step == cfg["steps_per_epoch"] - 1)
            if done or timeout:
                if done:
                    ep_rets.append(ep_ret)
                    buf.finish_path(last_val=0.0)
                else:
                    _, last_val, _ = agent.act(obs)
                    buf.finish_path(last_val=last_val)
                obs, _ = env.reset()
                ep_ret = 0.0

        last_portfolio_history = portfolio_hist
        data     = buf.get()
        obs_t    = torch.as_tensor(data["obs"],  dtype=torch.float32)
        act_t    = torch.as_tensor(data["act"],  dtype=torch.float32)
        adv_t    = torch.as_tensor(data["adv"],  dtype=torch.float32)
        ret_t    = torch.as_tensor(data["ret"],  dtype=torch.float32)
        logp_old = torch.as_tensor(data["logp"], dtype=torch.float32)

        kl = 0.0
        for _ in range(cfg["train_pi_iters"]):
            pi_opt.zero_grad()
            logp_new, _, _ = agent.evaluate(obs_t, act_t)
            ratio      = torch.exp(logp_new - logp_old)
            clip_adv   = torch.clamp(ratio, 1 - cfg["clip_ratio"], 1 + cfg["clip_ratio"]) * adv_t
            pi_loss    = -torch.min(ratio * adv_t, clip_adv).mean()
            kl         = float((logp_old - logp_new).detach().mean())
            # Symmetric early-stop: catches both positive and negative KL drift
            # (the approx KL `mean(logp_old - logp_new)` can be negative due to
            # sampling noise; the original `kl > threshold` test never fires
            # in that regime and lets the policy update for the full 100 iters).
            if abs(kl) > 1.5 * cfg["target_kl"]:
                break
            pi_loss.backward()
            pi_opt.step()

        for _ in range(cfg["train_v_iters"]):
            v_opt.zero_grad()
            _, v_new, _ = agent.evaluate(obs_t, act_t)
            v_loss = ((v_new - ret_t) ** 2).mean()
            v_loss.backward()
            v_opt.step()

        avg_ret = float(np.mean(ep_rets)) if ep_rets else 0.0
        line = (
            f"[{run_name}] Epoch {epoch+1:3d}/{cfg['epochs']} | "
            f"AvgEpRet {avg_ret:12.2f} | KL {kl:.4f} | "
            f"Time {time.time()-t0:.1f}s"
        )
        print(line)
        logs.append(line)

    torch.save(agent.state_dict(), save_path / f"{run_name}_agent.pth")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("\n".join(logs))
    print(f"[{run_name}] Model saved to {save_path}/{run_name}_agent.pth")

    return last_portfolio_history


def ppo_backtest(
    env,
    model_path: str,
    cfg: dict,
) -> list[float]:
    """Run a trained PPO agent on env and return portfolio value history."""
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent   = ActorCritic(obs_dim, act_dim, cfg.get("hidden_sizes", [512, 512]))
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    obs, _         = env.reset()
    portfolio_hist = []
    done           = False

    while not done:
        act, _, _ = agent.act(obs)
        obs, _, done, _, info = env.step(act)
        portfolio_hist.append(info.get("portfolio_value", 0.0))

    return portfolio_hist
