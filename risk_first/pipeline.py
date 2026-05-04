"""
Main pipeline entry point.

Usage:
  # Full run (train + evaluate):
  python -m risk_first.pipeline

  # Skip LLM signal generation (use neutral signals):
  python -m risk_first.pipeline --no-llm

  # Choose algorithm:
  python -m risk_first.pipeline --algo ppo

  # Custom config:
  python -m risk_first.pipeline --config risk_first/config.yaml
"""

import argparse
import os
import sys
import copy
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from risk_first.data.load_hf        import load_datasets as load_hf_datasets
from risk_first.data.download       import prepare_datasets, load_fnspid_sample, NASDAQ_TICKERS
from risk_first.signals.llm_signals import LLMSignalGenerator
from risk_first.env.trading_env     import StockTradingEnv
from risk_first.training.ppo        import ppo_train, ppo_backtest
from risk_first.training.cppo       import cppo_train, cppo_backtest
from risk_first.evaluation.metrics  import evaluate_all, get_benchmark, print_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_env(df: pd.DataFrame, cfg: dict, env_overrides: dict | None = None) -> StockTradingEnv:
    ec = cfg["env"].copy()
    if env_overrides:
        ec.update(env_overrides)
    return StockTradingEnv(
        df,
        initial_cash        = cfg["data"]["initial_cash"],
        hmax                = cfg["data"]["hmax"],
        reward_scaling      = ec.get("reward_scaling", 1e-4),
        use_confidence      = ec.get("use_confidence", True),
        use_reward_shaping  = ec.get("use_reward_shaping", True),
        use_circuit_breaker = ec.get("use_circuit_breaker", True),
        lambda_risk         = ec.get("lambda_risk", 0.02),
        cb_risk_threshold   = ec.get("cb_risk_threshold", 4.0),
        cb_sentiment_threshold = ec.get("cb_sentiment_threshold", 2.0),
    )


def run_pipeline(
    cfg_path: str = "risk_first/config.yaml",
    algo: str = "cppo",
    use_llm: bool = True,
    run_name: str | None = None,
    env_overrides: dict | None = None,
) -> dict[str, float]:
    load_dotenv()
    cfg = load_config(cfg_path)

    if run_name is None:
        run_name = f"{algo}{'_llm' if use_llm else '_baseline'}"

    cache_dir = Path("risk_first/cache")

    # ── Step 1 & 2: Load data ──────────────────────────────────────────────────
    # Priority: pre-computed HuggingFace dataset (benstaf/nasdaq_2013_2023)
    # which already contains llm_sentiment + llm_risk from DeepSeek-V3.
    # Fallback: generate signals via API (slow, needs DEEPINFRA_API_KEY).
    hf_train = cache_dir / "train.csv"
    hf_test  = cache_dir / "test.csv"

    if hf_train.exists() and hf_test.exists():
        print("[pipeline] Loading pre-processed HuggingFace data from cache...")
        train_df = pd.read_csv(hf_train, parse_dates=["date"])
        test_df  = pd.read_csv(hf_test,  parse_dates=["date"])
    elif use_llm:
        # Try to download the pre-computed HF dataset first
        try:
            train_df, test_df = load_hf_datasets(cache_dir=str(cache_dir))
        except Exception as e:
            print(f"[pipeline] HuggingFace download failed ({e}), falling back to API...")
            api_key = os.environ.get("DEEPINFRA_API_KEY")
            if not api_key:
                print("[pipeline] WARNING: no API key, using neutral signals.")
                train_df, test_df = prepare_datasets(cfg["data"], None, cache_dir=str(cache_dir))
            else:
                articles_df = load_fnspid_sample(NASDAQ_TICKERS, cfg["data"]["train_start"], cfg["data"]["test_end"])
                gen = LLMSignalGenerator(cfg["signals"])
                sig = gen.process_dataframe(articles_df, output_path=str(cache_dir / "signals.csv"))
                signals_df = pd.concat([articles_df[["date","tic"]].reset_index(drop=True), sig.reset_index(drop=True)], axis=1)
                train_df, test_df = prepare_datasets(cfg["data"], signals_df, cache_dir=str(cache_dir))
    else:
        train_df, test_df = prepare_datasets(cfg["data"], None, cache_dir=str(cache_dir))

    # ── Step 3: Train ──────────────────────────────────────────────────────────
    train_env = build_env(train_df, cfg, env_overrides)
    # During evaluation, disable reward shaping (it's a training artefact)
    test_env  = build_env(test_df, cfg, {**(env_overrides or {}), "use_reward_shaping": False})

    models_dir = "risk_first/models"
    logs_dir   = "risk_first/logs"

    if algo == "ppo":
        ppo_train(
            train_env, cfg["training"],
            save_dir=models_dir, run_name=run_name,
            log_path=f"{logs_dir}/{run_name}.log",
        )
        model_path    = f"{models_dir}/{run_name}_agent.pth"
        portfolio_hist = ppo_backtest(test_env, model_path, cfg["training"])
    else:
        cppo_train(
            train_env, cfg["training"], cfg["cppo"],
            save_dir=models_dir, run_name=run_name,
            log_path=f"{logs_dir}/{run_name}.log",
        )
        model_path    = f"{models_dir}/{run_name}_agent.pth"
        portfolio_hist = cppo_backtest(test_env, model_path, cfg["training"])

    # ── Step 4: Evaluate ───────────────────────────────────────────────────────
    benchmark_values = get_benchmark(
        cfg["eval"]["benchmark"],
        cfg["data"]["test_start"],
        cfg["data"]["test_end"],
        initial_cash=cfg["data"]["initial_cash"],
    )

    portfolio_arr = np.array(portfolio_hist)
    if portfolio_arr[0] == 0:
        portfolio_arr[0] = cfg["data"]["initial_cash"]

    metrics = evaluate_all(portfolio_arr, benchmark_values, alpha=cfg["eval"]["cvar_alpha"])
    print_metrics(metrics, label=run_name)

    # Save
    results_dir = Path("risk_first/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"model": run_name, **metrics}]).to_csv(
        results_dir / f"{run_name}_metrics.csv", index=False
    )
    # Persist equity curves for paper plots (one file per ablation config)
    np.save(results_dir / f"{run_name}_equity.npy",    portfolio_arr)
    np.save(results_dir / f"{run_name}_benchmark.npy", np.asarray(benchmark_values))

    return metrics


# Alias for run_ablation.py
def run_pipeline_with_cfg(cfg: dict, algo: str, use_llm: bool, run_name: str) -> dict[str, float]:
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name
    return run_pipeline(tmp_path, algo=algo, use_llm=use_llm, run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="risk_first/config.yaml")
    parser.add_argument("--algo",    default="cppo", choices=["ppo", "cppo"])
    parser.add_argument("--no-llm",  action="store_true")
    parser.add_argument("--name",    default=None)
    args = parser.parse_args()

    run_pipeline(
        cfg_path=args.config,
        algo=args.algo,
        use_llm=not args.no_llm,
        run_name=args.name,
    )
