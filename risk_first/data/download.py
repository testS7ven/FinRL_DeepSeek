"""
Data download and preparation pipeline.
Downloads OHLCV from Yahoo Finance, computes technical indicators,
merges with LLM signals, and splits into train/test.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

try:
    from stockstats import StockDataFrame
    _HAS_STOCKSTATS = True
except ImportError:
    _HAS_STOCKSTATS = False

# Subset of NASDAQ-100 with good FNSPID coverage
NASDAQ_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "COST", "NFLX", "ADP",  "AMD",  "QCOM", "INTC", "INTU", "AMAT",
    "ISRG", "MU",   "LRCX", "ADI",  "KLAC", "MRVL", "PANW", "SNPS",
    "CDNS", "FTNT", "MCHP", "NXPI", "ASML", "CRWD",
]

INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]


# ── OHLCV ─────────────────────────────────────────────────────────────────────

def download_ohlcv(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for tic in tickers:
        try:
            raw = yf.download(tic, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            raw = raw.reset_index()
            raw.columns = [
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in raw.columns
            ]
            raw["tic"] = tic
            frames.append(raw[["date", "open", "high", "low", "close", "volume", "tic"]])
        except Exception as e:
            print(f"[data] {tic}: download failed — {e}")

    if not frames:
        raise RuntimeError("No price data downloaded.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


def _add_indicators_stockstats(group: pd.DataFrame) -> pd.DataFrame:
    sdf = StockDataFrame.retype(group.copy())
    for ind in INDICATORS:
        try:
            group[ind] = sdf[ind].values
        except Exception:
            group[ind] = 0.0
    return group


def _add_indicators_manual(group: pd.DataFrame) -> pd.DataFrame:
    close = group["close"].values.astype(float)
    # MACD = EMA12 - EMA26
    def ema(x, n):
        s = pd.Series(x)
        return s.ewm(span=n, adjust=False).mean().values
    group["macd"]   = ema(close, 12) - ema(close, 26)
    # RSI 30
    delta = pd.Series(close).diff()
    gain  = delta.clip(lower=0).rolling(30).mean()
    loss  = (-delta.clip(upper=0)).rolling(30).mean()
    rs    = gain / (loss + 1e-8)
    group["rsi_30"] = 100 - 100 / (1 + rs.values)
    # CCI 30
    tp = (group["high"].values + group["low"].values + close) / 3
    tp_ma = pd.Series(tp).rolling(30).mean().values
    tp_std = pd.Series(tp).rolling(30).std().values
    group["cci_30"] = (tp - tp_ma) / (0.015 * tp_std + 1e-8)
    # DX 30 (simplified)
    group["dx_30"] = pd.Series(close).rolling(30).std().fillna(0).values
    return group


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    fn = _add_indicators_stockstats if _HAS_STOCKSTATS else _add_indicators_manual
    for _, group in df.groupby("tic"):
        group = group.sort_values("date").copy()
        group = fn(group)
        frames.append(group)
    result = pd.concat(frames, ignore_index=True)
    result[INDICATORS] = result[INDICATORS].fillna(0)
    return result.sort_values(["date", "tic"]).reset_index(drop=True)


# ── FNSPID ────────────────────────────────────────────────────────────────────

def load_fnspid_sample(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Stream FNSPID from HuggingFace and filter to tickers + date range."""
    try:
        from datasets import load_dataset
        print("[data] Streaming FNSPID from HuggingFace…")
        ds = load_dataset("Zihan1004/FNSPID", split="train", streaming=True)

        tickers_set = set(tickers)
        start_dt = pd.Timestamp(start)
        end_dt   = pd.Timestamp(end)
        rows = []

        for row in ds:
            tic  = row.get("Stock_symbol", "")
            date = pd.Timestamp(row.get("Date", "1900-01-01"))
            if tic in tickers_set and start_dt <= date <= end_dt:
                rows.append({
                    "date":        date.normalize(),
                    "tic":         tic,
                    "Lsa_summary": row.get("Lsa_summary", ""),
                })
            if len(rows) % 50_000 == 0 and rows:
                print(f"[data]   {len(rows):,} articles loaded…")

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        print(f"[data] FNSPID: {len(df):,} articles for {len(tickers)} tickers.")
        return df.sort_values(["date", "tic"]).reset_index(drop=True)

    except Exception as e:
        print(f"[data] FNSPID load failed: {e}")
        return pd.DataFrame(columns=["date", "tic", "Lsa_summary"])


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_signals(price_df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pre-computed LLM signals onto the price dataframe.
    signals_df must have columns: date, tic, sentiment, risk,
                                  sentiment_hat, risk_hat, confidence_risk.
    Missing entries → neutral values (3.0 / 0.0).
    """
    needed = {"date", "tic", "sentiment", "risk", "sentiment_hat", "risk_hat"}
    if not needed.issubset(set(signals_df.columns)):
        raise ValueError(f"signals_df is missing columns: {needed - set(signals_df.columns)}")

    agg = (
        signals_df
        .groupby(["date", "tic"])
        .agg(
            sentiment=("sentiment", "mean"),
            risk=("risk", "mean"),
            sentiment_hat=("sentiment_hat", "mean"),
            risk_hat=("risk_hat", "mean"),
            confidence_risk=("confidence_risk", "mean"),
        )
        .reset_index()
    )

    merged = price_df.merge(agg, on=["date", "tic"], how="left")
    merged["sentiment"]       = merged["sentiment"].fillna(3.0)
    merged["risk"]            = merged["risk"].fillna(3.0)
    merged["sentiment_hat"]   = merged["sentiment_hat"].fillna(3.0)
    merged["risk_hat"]        = merged["risk_hat"].fillna(3.0)
    merged["confidence_risk"] = merged["confidence_risk"].fillna(0.0)
    return merged


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare_datasets(
    cfg: dict,
    signals_df: pd.DataFrame | None = None,
    cache_dir: str = "risk_first/cache",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: download → indicators → merge signals → train/test split.
    Caches price data to avoid re-downloading.
    Returns (train_df, test_df).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    price_cache = cache / "prices.csv"

    start = min(cfg["train_start"], cfg["test_start"])
    end   = max(cfg["train_end"],   cfg["test_end"])

    if price_cache.exists():
        print("[data] Loading prices from cache…")
        df = pd.read_csv(price_cache, parse_dates=["date"])
    else:
        print("[data] Downloading OHLCV…")
        df = download_ohlcv(NASDAQ_TICKERS, start, end)
        df = add_technical_indicators(df)
        df.to_csv(price_cache, index=False)
        print(f"[data] Saved {len(df):,} rows to cache.")

    if signals_df is not None:
        df = merge_signals(df, signals_df)
    else:
        df["sentiment"]       = 3.0
        df["risk"]            = 3.0
        df["sentiment_hat"]   = 3.0
        df["risk_hat"]        = 3.0
        df["confidence_risk"] = 0.0

    train_df = df[(df["date"] >= cfg["train_start"]) & (df["date"] <= cfg["train_end"])].copy()
    test_df  = df[(df["date"] >= cfg["test_start"])  & (df["date"] <= cfg["test_end"])].copy()

    train_df.to_csv(cache / "train.csv", index=False)
    test_df.to_csv(cache  / "test.csv",  index=False)
    print(f"[data] Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")

    return train_df, test_df
