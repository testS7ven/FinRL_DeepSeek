"""
Module 1 — LLM signals with self-consistency confidence scoring.

For each article, generates:
  - sentiment      : raw score 1-5
  - risk           : raw score 1-5
  - confidence     : C_t ∈ [0, 1] (from variance across n_consistency passes)
  - sentiment_hat  : sentiment * C + 3 * (1 - C)  ← pulled toward neutral when uncertain
  - risk_hat       : risk * C + 3 * (1 - C)
"""

import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI
from pathlib import Path


_SENTIMENT_SYSTEM = (
    "You are a financial expert. "
    "Score each news headline from 1 (very negative) to 5 (very positive). "
    "Reply with ONLY comma-separated integers, one per article, nothing else."
)

_RISK_SYSTEM = (
    "You are a financial risk analyst. "
    "Score each news headline from 1 (very low risk) to 5 (very high risk). "
    "Score 3 means moderate/uncertain. "
    "Reply with ONLY comma-separated integers, one per article, nothing else."
)

_SENTIMENT_EXAMPLES = [
    {"role": "user",      "content": "Apple reports record revenue / Apple stock crashes 30% / Microsoft no change"},
    {"role": "assistant", "content": "5, 1, 3"},
]

_RISK_EXAMPLES = [
    {"role": "user",      "content": "Apple reports record revenue / Apple stock crashes 30% / Microsoft no change"},
    {"role": "assistant", "content": "1, 5, 3"},
]


def _confidence_weight(score: float, confidence: float) -> float:
    """Pull uncertain signal toward neutral (3)."""
    return score * confidence + 3.0 * (1.0 - confidence)


class LLMSignalGenerator:
    def __init__(self, cfg: dict):
        self.client = OpenAI(
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url=cfg["base_url"],
        )
        self.model = cfg["model"]
        self.n_consistency = cfg["n_consistency"]
        self.batch_size = cfg["batch_size"]
        self.cache_dir = Path(cfg["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _call(self, system: str, examples: list, articles: list[str], temp: float) -> list[int]:
        content = " / ".join(a[:300] for a in articles)
        messages = [{"role": "system", "content": system}] + examples + [{"role": "user", "content": content}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        scores = [int(x.strip()) for x in raw.split(",") if x.strip().lstrip("-").isdigit()]
        scores = [max(1, min(5, s)) for s in scores]
        if len(scores) != len(articles):
            scores = ([scores[0]] * len(articles) if scores else [3] * len(articles))
        return scores

    def _score_with_confidence(
        self, system: str, examples: list, articles: list[str]
    ) -> tuple[list[float], list[float]]:
        """
        Run n_consistency passes with slight temperature variation.
        Returns (mean_scores, confidences).
        Confidence = 1 - std/2, so std=0 → C=1, std≥2 → C≤0.
        """
        all_scores: list[list[int]] = []
        temps = np.linspace(0.0, 0.4, self.n_consistency)
        for temp in temps:
            try:
                scores = self._call(system, examples, articles, float(temp))
                all_scores.append(scores)
                time.sleep(0.3)
            except Exception as e:
                print(f"[signals] API error: {e}")
                continue

        if not all_scores:
            return [3.0] * len(articles), [0.0] * len(articles)

        arr = np.array(all_scores, dtype=float)  # (n_calls, n_articles)
        means = arr.mean(axis=0).tolist()
        stds = arr.std(axis=0).tolist()
        confidences = [float(max(0.0, 1.0 - s / 2.0)) for s in stds]
        return means, confidences

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "Lsa_summary",
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Score all articles in df. Adds 6 columns:
          sentiment, confidence_sentiment, sentiment_hat,
          risk,      confidence_risk,      risk_hat

        Supports resume: skips rows already in output_path.
        """
        out_path = Path(output_path) if output_path else self.cache_dir / "signals.csv"

        done_rows = 0
        if out_path.exists():
            done_df = pd.read_csv(out_path)
            done_rows = len(done_df)
            print(f"[signals] Resuming from row {done_rows}/{len(df)}")
        else:
            done_df = pd.DataFrame()

        texts = df[text_col].fillna("no information").tolist()
        new_results: list[dict] = []

        for i in range(done_rows, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                s_scores, s_conf = self._score_with_confidence(_SENTIMENT_SYSTEM, _SENTIMENT_EXAMPLES, batch)
                r_scores, r_conf = self._score_with_confidence(_RISK_SYSTEM, _RISK_EXAMPLES, batch)
            except Exception as e:
                print(f"[signals] Batch {i} failed: {e}")
                s_scores = [3.0] * len(batch)
                s_conf   = [0.0] * len(batch)
                r_scores = [3.0] * len(batch)
                r_conf   = [0.0] * len(batch)

            for j in range(len(batch)):
                new_results.append({
                    "sentiment":          s_scores[j],
                    "confidence_sentiment": s_conf[j],
                    "sentiment_hat":      _confidence_weight(s_scores[j], s_conf[j]),
                    "risk":               r_scores[j],
                    "confidence_risk":    r_conf[j],
                    "risk_hat":           _confidence_weight(r_scores[j], r_conf[j]),
                })

            # Checkpoint every 50 batches
            if len(new_results) % (50 * self.batch_size) == 0:
                self._save(done_df, new_results, out_path)
                print(f"[signals] Checkpoint: {done_rows + len(new_results)}/{len(texts)}")

        return self._save(done_df, new_results, out_path)

    @staticmethod
    def _save(done_df: pd.DataFrame, new_results: list[dict], out_path: Path) -> pd.DataFrame:
        new_df = pd.DataFrame(new_results)
        combined = pd.concat([done_df, new_df], ignore_index=True)
        combined.to_csv(out_path, index=False)
        return combined
