"""Build the ball-by-ball dataset Section 4 (PySpark) uses.

Run once:   python scripts/prepare_spark_data.py

Produces `data/odi_deliveries.parquet` — one row per **delivery** (about 1.35
million of them) from the same Cricsheet men's ODI archive Section 7 uses.

Why a separate, bigger file? Because Section 4 is about *scale*. The 25k-row
innings table from Section 7 fits comfortably in pandas; a million-plus rows is
where you start to feel the difference — small enough to still run on a laptop,
big enough that the lesson isn't a lie.

Parquet, not CSV, on purpose: it's columnar, compressed, and carries its own
schema. It is what you'd actually use in a Spark job, and the file is ~10x
smaller than the equivalent CSV.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URL = "https://cricsheet.org/downloads/odis_male_csv2.zip"
OUT = DATA / "odi_deliveries.parquet"

COLS = ["match_id", "season", "start_date", "venue", "innings", "ball",
        "batting_team", "bowling_team", "striker", "bowler", "runs_off_bat",
        "extras", "wides", "wicket_type", "player_dismissed"]


def main() -> int:
    DATA.mkdir(exist_ok=True)
    print(f"downloading {URL} …")
    with urllib.request.urlopen(URL) as r:  # noqa: S310
        zf = zipfile.ZipFile(io.BytesIO(r.read()))

    names = [n for n in zf.namelist()
             if n.endswith(".csv") and not n.endswith("_info.csv")]
    print(f"parsing {len(names)} matches …")
    frames = []
    for i, n in enumerate(names):
        if i % 500 == 0:
            print(f"  {i}/{len(names)}")
        with zf.open(n) as fh:
            frames.append(pd.read_csv(fh, usecols=lambda c: c in COLS,
                                      low_memory=False))
    df = pd.concat(frames, ignore_index=True)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["year"] = df["start_date"].dt.year
    df["runs_off_bat"] = df["runs_off_bat"].fillna(0).astype("int16")
    df["extras"] = df["extras"].fillna(0).astype("int16")
    df["is_wide"] = df["wides"].notna().astype("int8")
    df["is_wicket"] = df["player_dismissed"].notna().astype("int8")
    df["innings"] = df["innings"].astype("int8")
    df["over"] = df["ball"].astype(float).apply(int).astype("int16")

    keep = ["match_id", "year", "venue", "innings", "over", "ball",
            "batting_team", "bowling_team", "striker", "bowler",
            "runs_off_bat", "extras", "is_wide", "is_wicket", "wicket_type"]
    out = df[keep]
    out.to_parquet(OUT, index=False, compression="snappy")

    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}")
    print(f"  {len(out):,} deliveries, {len(out.columns)} columns")
    print(f"  {size_mb:.1f} MB on disk (parquet)")
    print(f"  in-memory as pandas: "
          f"{out.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"  years {out.year.min()}–{out.year.max()}, "
          f"{out.match_id.nunique():,} matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
