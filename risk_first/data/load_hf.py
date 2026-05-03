"""
Loader for the pre-computed benstaf/nasdaq_2013_2023 dataset.
Downloads once, caches locally, and prepares train/test DataFrames
in the exact format expected by StockTradingEnv.

Columns produced:
  date, tic, close, high, low, open, volume,
  macd, rsi_30, cci_30, dx_30,
  sentiment, risk, sentiment_hat, risk_hat, confidence_risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download


HF_REPO   = "benstaf/nasdaq_2013_2023"
HF_FILES  = {
    "train": "train_data_deepseek_risk_2013_2018.csv",
    "test":  "trade_data_deepseek_risk_2019_2023.csv",
}
CACHE_DIR = Path("risk_first/cache/hf_raw")


def _download_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for split, fname in HF_FILES.items():
        local = CACHE_DIR / fname
        if not local.exists():
            print(f"[load_hf] Downloading {fname} from HuggingFace...")
            path = hf_hub_download(
                repo_id=HF_REPO, filename=fname,
                repo_type="dataset", local_dir=str(CACHE_DIR),
            )
        else:
            path = str(local)
            print(f"[load_hf] Using cached {fname}")
        dfs[split] = pd.read_csv(path, parse_dates=["date"])
    return dfs["train"], dfs["test"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw HuggingFace DataFrame to the format
    expected by StockTradingEnv:
      - rename llm_sentiment → sentiment, llm_risk → risk
      - fill NaN signals with neutral (3.0)
      - add sentiment_hat = risk_hat = signal (confidence=1 since single pass)
      - keep only the columns the env needs
    """
    df = df.copy()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    # Rename
    df = df.rename(columns={"llm_sentiment": "sentiment", "llm_risk": "risk"})

    # Fill missing signals with neutral
    df["sentiment"] = df["sentiment"].fillna(3.0).clip(1, 5)
    df["risk"]      = df["risk"].fillna(3.0).clip(1, 5)

    # sentiment=0 is out-of-scale in the original data → treat as neutral
    df.loc[df["sentiment"] == 0, "sentiment"] = 3.0

    # confidence_risk = 0 when no article (NaN filled), 1 when signal exists
    # We don't have self-consistency here so confidence = 0.7 as a conservative default
    df["confidence_risk"]  = np.where(df["risk"] == 3.0, 0.0, 0.7)

    # Weighted signals (Module 1 formula with fixed confidence)
    df["sentiment_hat"] = df["sentiment"] * df["confidence_risk"] + 3.0 * (1 - df["confidence_risk"])
    df["risk_hat"]      = df["risk"]      * df["confidence_risk"] + 3.0 * (1 - df["confidence_risk"])

    keep = [
        "date", "tic", "close", "high", "low", "open", "volume",
        "macd", "rsi_30", "cci_30", "dx_30",
        "sentiment", "risk", "sentiment_hat", "risk_hat", "confidence_risk",
    ]
    df = df[[c for c in keep if c in df.columns]]
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def load_datasets(cache_dir: str = "risk_first/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, test_df) ready to pass to StockTradingEnv.
    Saves processed files to cache_dir for reuse.
    """
    cache = Path(cache_dir)
    train_cache = cache / "train.csv"
    test_cache  = cache / "test.csv"

    if train_cache.exists() and test_cache.exists():
        print("[load_hf] Loading processed data from cache...")
        train_df = pd.read_csv(train_cache, parse_dates=["date"])
        test_df  = pd.read_csv(test_cache,  parse_dates=["date"])
        return train_df, test_df

    train_raw, test_raw = _download_raw()
    train_df = _prepare(train_raw)
    test_df  = _prepare(test_raw)

    cache.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_cache, index=False)
    test_df.to_csv(test_cache,   index=False)

    print(f"[load_hf] Train: {len(train_df):,} rows | {train_df['tic'].nunique()} tickers")
    print(f"[load_hf] Test:  {len(test_df):,} rows  | {test_df['tic'].nunique()} tickers")
    sent_pct = (train_df['sentiment'] != 3.0).mean() * 100
    risk_pct  = (train_df['risk'] != 3.0).mean() * 100
    print(f"[load_hf] Signal coverage — sentiment: {sent_pct:.1f}% | risk: {risk_pct:.1f}%")

    return train_df, test_df
