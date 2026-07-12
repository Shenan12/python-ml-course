"""The ODI batting dataset used throughout Section 7 (DeepHit).

One row per batter-innings, built from Cricsheet's ball-by-ball men's ODI data
by `scripts/prepare_cricket_data.py`.

    duration = balls faced      ("time")
    event    = 1 dismissed, 0 NOT OUT   (right-censored — real censoring!)

Everything here is deliberately small and cheap: the whole point is that these
models train in seconds on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[1] / "data" / "odi_batting_innings.csv"

NUMERIC = [
    "career_avg",                   # runs per dismissal, before this innings
    "career_sr",                    # strike rate, before this innings
    "career_balls_per_dismissal",   # survival-flavoured form measure
    "career_innings",               # experience
    "batting_position",             # 1 = opener
    "entry_over",                   # over in which they came to the crease
    "runs_at_entry",                # team score when they walked in
    "wickets_at_entry",             # wickets already down
]
BINARY = ["is_chasing"]

# the teams that play enough ODIs to be worth their own column
TOP_TEAMS = ["India", "Australia", "England", "Pakistan", "New Zealand",
             "South Africa", "Sri Lanka", "Bangladesh", "West Indies",
             "Afghanistan"]

# temporal split: train on the past, test on the future — the only honest way
# to judge a model you'd actually deploy
TRAIN_MAX_YEAR = 2019
VAL_MAX_YEAR = 2022


def load_raw() -> pd.DataFrame:
    if not CSV.exists():
        raise FileNotFoundError(
            f"{CSV} is missing. Run:  python scripts/prepare_cricket_data.py")
    return pd.read_csv(CSV, parse_dates=["start_date"])


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering — every step is explained on the 'Cricket Data &
    Feature Engineering' page."""
    out = df.copy()

    # one-hot the opposition bowling attack (Section 3: nominal categories
    # must never be encoded as ordered integers)
    opp = out["bowling_team"].where(out["bowling_team"].isin(TOP_TEAMS),
                                    "Other")
    out = pd.concat([out, pd.get_dummies(opp, prefix="vs", dtype=int)], axis=1)

    # a batter arriving at 15/3 faces a very different innings from one
    # arriving at 150/1 — give the model the *rate*, not just the raw score
    out["run_rate_at_entry"] = out["runs_at_entry"] / np.maximum(
        out["entry_over"], 1)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    vs_cols = sorted(c for c in df.columns if c.startswith("vs_"))
    return NUMERIC + ["run_rate_at_entry"] + BINARY + vs_cols


def splits(df: pd.DataFrame | None = None):
    """Train / validation / test, split by TIME (not at random).

    Returns three DataFrames. Splitting cricket innings at random would let the
    model learn from 2024 matches to predict 2018 ones — and, worse, our career
    features are cumulative, so a random split leaks a batter's future form
    into his past. Time-ordered splits are the honest choice.
    """
    if df is None:
        df = engineer(load_raw())
    train = df[df["year"] <= TRAIN_MAX_YEAR]
    val = df[(df["year"] > TRAIN_MAX_YEAR) & (df["year"] <= VAL_MAX_YEAR)]
    test = df[df["year"] > VAL_MAX_YEAR]
    return train, val, test


def xy(df: pd.DataFrame, cols: list[str], mu=None, sd=None):
    """Standardized feature matrix + (duration, event) targets."""
    X = df[cols].astype("float32").to_numpy()
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
    Xs = ((X - mu) / sd).astype("float32")
    dur = df["balls_faced"].to_numpy().astype("float32")
    ev = df["event"].to_numpy().astype("float32")
    return Xs, dur, ev, mu, sd
