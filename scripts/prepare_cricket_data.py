"""Turn Cricsheet's ball-by-ball ODI data into a survival-analysis dataset.

Run once:   python scripts/prepare_cricket_data.py

The output (`data/odi_batting_innings.csv`) ships with the course, so you do
not need to run this — it's here so you can see exactly how the data was made,
and rebuild it if you want a different era or filter.

The survival framing
--------------------
One row = one batter's innings.

    duration : balls faced          (our "time")
    event    : 1 = dismissed, 0 = NOT OUT  (right-censored!)

A "not out" batter is a genuinely censored observation: the innings ended (the
overs ran out, or the team won) while they were still batting. We know they
survived *at least* that many balls, but not how long they would have lasted.
That is precisely the censoring structure survival analysis exists for — and
it is real here, not simulated.

Source: https://cricsheet.org (men's ODIs, CC BY 4.0).
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://cricsheet.org/downloads/odis_male_csv2.zip"
OUT = DATA / "odi_batting_innings.csv"

# keep the dataset modest so models train in seconds on a laptop
MIN_SEASON = 2010
MIN_PRIOR_INNINGS = 5      # a batter needs some history for career features


def download() -> zipfile.ZipFile:
    print(f"downloading {URL} …")
    with urllib.request.urlopen(URL) as r:  # noqa: S310
        return zipfile.ZipFile(io.BytesIO(r.read()))


def load_balls(zf: zipfile.ZipFile) -> pd.DataFrame:
    names = [n for n in zf.namelist()
             if n.endswith(".csv") and not n.endswith("_info.csv")]
    print(f"parsing {len(names)} matches …")
    frames = []
    cols = ["match_id", "start_date", "innings", "ball", "batting_team",
            "bowling_team", "striker", "non_striker", "runs_off_bat", "wides",
            "wicket_type", "player_dismissed", "venue"]
    for i, n in enumerate(names):
        if i % 500 == 0:
            print(f"  {i}/{len(names)}")
        with zf.open(n) as fh:
            frames.append(pd.read_csv(fh, usecols=lambda c: c in cols,
                                      low_memory=False))
    return pd.concat(frames, ignore_index=True)


def build_innings(balls: pd.DataFrame) -> pd.DataFrame:
    balls["start_date"] = pd.to_datetime(balls["start_date"], errors="coerce")
    balls = balls[balls["innings"].isin([1, 2])].copy()
    balls["year"] = balls["start_date"].dt.year
    balls = balls[balls["year"] >= MIN_SEASON].copy()

    # a WIDE is not a ball faced by the batter; everything else is
    balls["is_ball_faced"] = balls["wides"].isna() | (balls["wides"] == 0)
    balls["runs_off_bat"] = balls["runs_off_bat"].fillna(0)

    key = ["match_id", "innings"]

    # running team state BEFORE each delivery (for "situation at entry")
    balls = balls.sort_values(key + ["ball"])
    total_runs = balls["runs_off_bat"].fillna(0)
    balls["team_runs_before"] = (total_runs.groupby(
        [balls["match_id"], balls["innings"]]).cumsum() - total_runs)
    is_wkt = balls["player_dismissed"].notna().astype(int)
    balls["wkts_before"] = (is_wkt.groupby(
        [balls["match_id"], balls["innings"]]).cumsum() - is_wkt)

    # --- per (match, innings, striker): balls faced and runs ---
    faced = balls[balls["is_ball_faced"]]
    agg = (faced.groupby(key + ["striker"], as_index=False)
           .agg(balls_faced=("ball", "size"),
                runs=("runs_off_bat", "sum"),
                entry_ball=("ball", "min"),
                start_date=("start_date", "first"),
                year=("year", "first"),
                batting_team=("batting_team", "first"),
                bowling_team=("bowling_team", "first"),
                venue=("venue", "first")))
    agg = agg.rename(columns={"striker": "batter"})

    # situation when the batter first faced a ball
    entry = (faced.sort_values("ball").groupby(key + ["striker"], as_index=False)
             .first()[key + ["striker", "team_runs_before", "wkts_before"]]
             .rename(columns={"striker": "batter",
                              "team_runs_before": "runs_at_entry",
                              "wkts_before": "wickets_at_entry"}))
    agg = agg.merge(entry, on=key + ["batter"], how="left")

    # --- was this batter dismissed in this innings? ---
    # (a batter can be run out while at the NON-striker's end, so we look at
    #  player_dismissed directly rather than assuming they were on strike)
    outs = balls[balls["player_dismissed"].notna()][
        key + ["player_dismissed", "wicket_type"]].copy()
    outs = outs.rename(columns={"player_dismissed": "batter"})
    # "retired hurt" means they left without being out -> still censored
    outs["event"] = (~outs["wicket_type"].isin(
        ["retired hurt", "retired not out"])).astype(int)
    outs = (outs.sort_values("event")
            .drop_duplicates(subset=key + ["batter"], keep="last"))
    agg = agg.merge(outs[key + ["batter", "event", "wicket_type"]],
                    on=key + ["batter"], how="left")
    agg["event"] = agg["event"].fillna(0).astype(int)   # never dismissed = NOT OUT

    # batting position = order of first ball faced within the innings
    agg = agg.sort_values(key + ["entry_ball"])
    agg["batting_position"] = agg.groupby(key).cumcount() + 1

    agg = agg[agg["balls_faced"] >= 1].copy()
    return agg


def add_career_features(df: pd.DataFrame) -> pd.DataFrame:
    """Career form BEFORE this innings — expanding, shifted by one.

    This is the anti-leakage rule from Section 2, applied to time: a feature
    for an innings played in 2015 may only use innings played BEFORE it. Using
    a batter's full-career average would leak the future into the past and
    flatter the model enormously.
    """
    df = df.sort_values(["batter", "start_date", "match_id"]).copy()
    g = df.groupby("batter")

    prior_inns = g.cumcount()
    prior_runs = g["runs"].cumsum() - df["runs"]
    prior_balls = g["balls_faced"].cumsum() - df["balls_faced"]
    prior_outs = g["event"].cumsum() - df["event"]

    df["career_innings"] = prior_inns
    df["career_avg"] = prior_runs / prior_outs.replace(0, np.nan)
    df["career_avg"] = df["career_avg"].fillna(prior_runs)   # never out yet
    df["career_sr"] = 100 * prior_runs / prior_balls.replace(0, np.nan)
    df["career_balls_per_dismissal"] = (
        prior_balls / prior_outs.replace(0, np.nan))
    df["career_balls_per_dismissal"] = df["career_balls_per_dismissal"].fillna(
        prior_balls)

    df = df[df["career_innings"] >= MIN_PRIOR_INNINGS].copy()
    df["career_avg"] = df["career_avg"].clip(0, 80)
    df["career_sr"] = df["career_sr"].clip(20, 200)
    df["career_balls_per_dismissal"] = df["career_balls_per_dismissal"].clip(
        0, 120)
    return df


def main() -> int:
    DATA.mkdir(exist_ok=True)
    zf = download()
    balls = load_balls(zf)
    print(f"{len(balls):,} deliveries")

    inns = build_innings(balls)
    print(f"{len(inns):,} batter-innings")

    inns = add_career_features(inns)
    inns["is_chasing"] = (inns["innings"] == 2).astype(int)
    inns["entry_over"] = (inns["entry_ball"].astype(float)
                          .apply(np.floor).astype(int))

    keep = ["match_id", "innings", "start_date", "year", "batter",
            "batting_team", "bowling_team", "venue", "balls_faced", "runs",
            "event", "wicket_type", "batting_position", "entry_over",
            "runs_at_entry", "wickets_at_entry", "is_chasing",
            "career_innings", "career_avg", "career_sr",
            "career_balls_per_dismissal"]
    out = inns[keep].sort_values(["start_date", "match_id", "innings",
                                  "batting_position"])
    out.to_csv(OUT, index=False)

    print(f"\nwrote {OUT}  ({len(out):,} rows)")
    print(f"  dismissed (event=1): {out['event'].sum():,} "
          f"({out['event'].mean():.1%})")
    print(f"  NOT OUT  (censored): {(1 - out['event']).sum():,} "
          f"({1 - out['event'].mean():.1%})")
    print(f"  balls faced: median {out['balls_faced'].median():.0f}, "
          f"max {out['balls_faced'].max()}")
    print(f"  seasons {out['year'].min()}–{out['year'].max()}, "
          f"{out['batter'].nunique():,} batters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
