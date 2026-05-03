"""
Ablation study — runs configurations A → F and produces a comparison table.

Configurations:
  A — PPO baseline           (no LLM, no CVaR)
  B — PPO + LLM signals      (raw signals, no CVaR)
  C — CPPO baseline          (CVaR only, no LLM)
  D — CPPO + Confidence      (CVaR + confidence-weighted signals)
  E — CPPO + Conf + Shaping  (+ reward shaping Module 2)
  F — Full model             (+ circuit-breaker Module 3)

Usage:
  python -m risk_first.run_ablation
  python -m risk_first.run_ablation --skip A B   # skip specific configs
"""

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

from risk_first.pipeline          import run_pipeline_with_cfg, load_config
from risk_first.evaluation.metrics import metrics_to_df, print_metrics

CONFIGS = {
    "A_PPO_baseline": {
        "algo": "ppo",
        "use_llm": False,
        "env_overrides": {
            "use_confidence":      False,
            "use_reward_shaping":  False,
            "use_circuit_breaker": False,
        },
    },
    "B_PPO_LLM": {
        "algo": "ppo",
        "use_llm": True,
        "env_overrides": {
            "use_confidence":      False,
            "use_reward_shaping":  False,
            "use_circuit_breaker": False,
        },
    },
    "C_CPPO_baseline": {
        "algo": "cppo",
        "use_llm": False,
        "env_overrides": {
            "use_confidence":      False,
            "use_reward_shaping":  False,
            "use_circuit_breaker": False,
        },
    },
    "D_CPPO_confidence": {
        "algo": "cppo",
        "use_llm": True,
        "env_overrides": {
            "use_confidence":      True,
            "use_reward_shaping":  False,
            "use_circuit_breaker": False,
        },
    },
    "E_CPPO_shaping": {
        "algo": "cppo",
        "use_llm": True,
        "env_overrides": {
            "use_confidence":      True,
            "use_reward_shaping":  True,
            "use_circuit_breaker": False,
        },
    },
    "F_full_model": {
        "algo": "cppo",
        "use_llm": True,
        "env_overrides": {
            "use_confidence":      True,
            "use_reward_shaping":  True,
            "use_circuit_breaker": True,
        },
    },
}


def run_ablation(
    cfg_path: str = "risk_first/config.yaml",
    skip: list[str] | None = None,
):
    load_dotenv()
    base_cfg = load_config(cfg_path)
    skip     = [s.upper() for s in (skip or [])]

    results: dict[str, dict] = {}
    out_dir = Path("risk_first/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from previous partial run
    partial_path = out_dir / "ablation_results.json"
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        print(f"[ablation] Resuming — {len(results)} configs already done.")

    for name, spec in CONFIGS.items():
        short = name.split("_")[0]
        if short in skip:
            print(f"[ablation] Skipping {name}")
            continue
        if name in results:
            print(f"[ablation] {name} already done, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")

        cfg = copy.deepcopy(base_cfg)
        for k, v in spec["env_overrides"].items():
            cfg["env"][k] = v

        try:
            metrics = run_pipeline_with_cfg(
                cfg=cfg,
                algo=spec["algo"],
                use_llm=spec["use_llm"],
                run_name=name,
            )
            results[name] = metrics
        except Exception as e:
            print(f"[ablation] ERROR in {name}: {e}")
            results[name] = {"error": str(e)}

        # Save after each config
        with open(partial_path, "w") as f:
            json.dump(results, f, indent=2)

    # ── Final table ────────────────────────────────────────────────────────────
    clean = {k: v for k, v in results.items() if "error" not in v}
    if clean:
        df = metrics_to_df(clean)
        print("\n\n")
        print(df.to_string(float_format=lambda x: f"{x:.4f}"))
        df.to_csv(out_dir / "ablation_summary.csv")
        print(f"\nResults saved to {out_dir}/ablation_summary.csv")

    errors = {k: v for k, v in results.items() if "error" in v}
    if errors:
        print(f"\nFailed configs: {list(errors.keys())}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="risk_first/config.yaml")
    parser.add_argument("--skip", nargs="*", default=[], help="Config prefixes to skip, e.g. A B")
    args = parser.parse_args()

    run_ablation(cfg_path=args.config, skip=args.skip)
