"""Precompute the course's expensive results, once.

Some pages (Section 6, DeepSurv) train real neural networks. Training them on
every page load would make the app crawl on a laptop, so this script runs them
ONCE and saves the results into `artifacts/`. The pages then load those results
instantly, and every one of them still offers a ▶ Run live button if you want
to watch the training happen for real.

Nothing here is hand-written data: every number saved is the output of actually
executing the code the pages display.

Usage
-----
    python build_artifacts.py            # build anything that's missing
    python build_artifacts.py --force    # rebuild everything from scratch
    python build_artifacts.py --only d7  # rebuild just one group

Takes roughly 10 minutes from cold. You only ever need to run it if you edit
one of the heavy code snippets (the app will tell you if you have).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from utils import compat  # noqa: E402, F401  (before pycox/torchtuples)
from sections.section4_pyspark import _experiments as S  # noqa: E402
from sections.section6_deepsurv import _experiments as X  # noqa: E402
from sections.section7_deephit import _experiments as H  # noqa: E402
from utils import artifacts  # noqa: E402


def _run_snippet(code: str) -> str:
    """Execute a teaching snippet exactly as the sandbox would, and capture
    what it printed. This is what makes the saved output trustworthy: it is
    produced by running the very string the page renders."""
    namespace = {"__name__": "__build__"}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, "<snippet>", "exec"), namespace)  # noqa: S102
    return buffer.getvalue()


def build_snippets(force: bool) -> None:
    for key, code in X.SNIPPETS.items():
        name = f"snippet_{key}"
        existing = artifacts.load(name)
        if (not force and existing
                and existing.get("code_hash") == artifacts.code_hash(code)):
            print(f"  · {key:14} up to date")
            continue
        t0 = time.time()
        stdout = _run_snippet(code)
        artifacts.save(name, {"stdout": stdout,
                              "code_hash": artifacts.code_hash(code)}, seed=0)
        print(f"  ✓ {key:14} {time.time() - t0:6.1f}s")
        for line in stdout.strip().splitlines()[-2:]:
            print(f"      {line}")


def build_d5(force: bool) -> None:
    if not force and artifacts.exists("d5_standardization"):
        print("  · d5_standardization up to date")
        return
    t0 = time.time()
    res = X.standardization_experiment()
    artifacts.save("d5_standardization", res, seed=0)
    print(f"  ✓ d5_standardization {time.time() - t0:6.1f}s")
    for scaling in ["raw", "standardized"]:
        for opt in ["SGD", "Adam"]:
            k = f"{scaling}_{opt}"
            c = res[f"{k}_cindex"]
            verdict = ("DIVERGED (NaN)" if res[f"{k}_diverged"]
                       else f"C={c:.3f}")
            print(f"      {scaling:13} + {opt:4}: {verdict}")


def build_d6(force: bool) -> None:
    if not force and artifacts.exists("d6_recommender"):
        print("  · d6_recommender up to date")
        return
    t0 = time.time()
    res = X.recommender_experiment()
    artifacts.save("d6_recommender", res, seed=0)
    print(f"  ✓ d6_recommender {time.time() - t0:6.1f}s  "
          f"cox gamma={res['gamma']:+.3f}  "
          f"median agree={res['median_agree']:.2f} vs "
          f"disagree={res['median_disagree']:.2f}")


def build_d7(force: bool) -> None:
    combos = [(r, n, es) for r in X.D7_RISKS for n in X.D7_TRAIN_SIZES
              for es in X.D7_EARLY_STOP]
    print(f"  ({len(combos)} slider combinations — every one is a real fit)")
    for risk, n, early_stop in combos:
        name = f"d7_{risk}_{n}_{'es' if early_stop else 'full'}"
        if not force and artifacts.exists(name):
            print(f"  · {name:26} up to date")
            continue
        t0 = time.time()
        res = X.simulation_experiment(risk, n, early_stop)
        artifacts.save(name, res, seed=0)
        print(f"  ✓ {name:26} {time.time() - t0:6.1f}s  "
              f"Cox {res['c_cox']:.3f} | DeepSurv {res['c_ds']:.3f} "
              f"({res['epochs_run']} epochs)")


def build_d8(force: bool) -> None:
    if not force and artifacts.exists("d8_random_search"):
        print("  · d8_random_search up to date")
        return
    t0 = time.time()
    res = X.random_search_experiment(X.D8_MAX_TRIALS)
    artifacts.save("d8_random_search", res, seed=7)
    df = X.trials_frame(res, X.D8_MAX_TRIALS)
    best = df.loc[df["validation C-index"].idxmax()]
    print(f"  ✓ d8_random_search {time.time() - t0:6.1f}s  "
          f"{len(df)} trials, best val C={best['validation C-index']:.4f} "
          f"(test {best['test C-index']:.4f})")


def _build_one(name: str, fn, force: bool, summary=None) -> None:
    if not force and artifacts.exists(name):
        print(f"  · {name:22} up to date")
        return
    t0 = time.time()
    res = fn()
    artifacts.save(name, res, seed=0)
    line = f"  ✓ {name:22} {time.time() - t0:6.1f}s"
    if summary:
        line += "  " + summary(res)
    print(line)


def build_h1(force: bool) -> None:
    _build_one("h1_ph_violation", H.ph_violation_experiment, force,
               lambda r: (f"n={r['n']:,}  censored={r['censor_rate']:.1%}  "
                          f"Cox C={r['cox_cindex']:.3f}"))


def build_h2(force: bool) -> None:
    _build_one("h2_leakage", H.leakage_experiment, force,
               lambda r: (f"time-split C={r['c_time_split']:.4f}  "
                          f"random-split C={r['c_random_split']:.4f}"))


def build_h3(force: bool) -> None:
    _build_one("h3_binning", H.binning_experiment, force)


def build_h4(force: bool) -> None:
    _build_one("h4_pmf", H.pmf_experiment, force)


def build_h5(force: bool) -> None:
    _build_one("h5_alpha", H.alpha_experiment, force)


def build_h7(force: bool) -> None:
    _build_one("h7_head_to_head", H.head_to_head_experiment, force,
               lambda r: (f"Cox {r['c_cox']:.4f} | DeepSurv {r['c_ds']:.4f} | "
                          f"DeepHit {r['c_dh']:.4f}"))


def build_snippets7(force: bool) -> None:
    for key, code in H.SNIPPETS.items():
        name = f"snippet_{key}"
        existing = artifacts.load(name)
        if (not force and existing
                and existing.get("code_hash") == artifacts.code_hash(code)):
            print(f"  · {key:14} up to date")
            continue
        t0 = time.time()
        stdout = _run_snippet(code)
        artifacts.save(name, {"stdout": stdout,
                              "code_hash": artifacts.code_hash(code)}, seed=0)
        print(f"  ✓ {key:14} {time.time() - t0:6.1f}s")


def build_spark(force: bool) -> None:
    """Section 4. Each of these boots a real SparkSession (~15 s each)."""
    _build_one("s1_scale", S.scale_experiment, force,
               lambda r: (f"{r['n_rows']:,} rows · {r['disk_mb']:.1f} MB disk "
                          f"→ {r['pandas_mem_mb']:.0f} MB in pandas"))
    _build_one("s3_lazy", S.lazy_experiment, force,
               lambda r: (f"transforms {r['t_transform_total']*1000:.0f} ms · "
                          f"action {r['t_action']:.2f} s"))
    _build_one("s4_operations", S.operations_experiment, force,
               lambda r: f"DSL == SQL: {r['dsl_sql_identical']}")
    _build_one("s6_partitions", S.partition_experiment, force,
               lambda r: (f"{r['default_partitions']} partitions · narrow "
                          f"{r['t_narrow']:.2f}s vs wide {r['t_wide']:.2f}s"))
    _build_one("s7_mllib", S.mllib_experiment, force,
               lambda r: f"train {r['n_train']:,} rows")
    _build_one("s8_comparison", S.comparison_experiment, force)

    # the teaching snippets — each boots a real SparkSession
    for key, code in S.SNIPPETS.items():
        name = f"snippet_{key}"
        existing = artifacts.load(name)
        if (not force and existing
                and existing.get("code_hash") == artifacts.code_hash(code)):
            print(f"  · {key:22} up to date")
            continue
        t0 = time.time()
        stdout = _run_snippet(code)
        artifacts.save(name, {"stdout": stdout,
                              "code_hash": artifacts.code_hash(code)}, seed=0)
        print(f"  ✓ {key:22} {time.time() - t0:6.1f}s")


GROUPS = {
    "spark": build_spark,
    "snippets": build_snippets,
    "d5": build_d5,
    "d6": build_d6,
    "d7": build_d7,
    "d8": build_d8,
    "snippets7": build_snippets7,
    "h1": build_h1,
    "h2": build_h2,
    "h3": build_h3,
    "h4": build_h4,
    "h5": build_h5,
    "h7": build_h7,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if an artifact already exists")
    parser.add_argument("--only", choices=sorted(GROUPS),
                        help="build only one group")
    args = parser.parse_args()

    groups = {args.only: GROUPS[args.only]} if args.only else GROUPS
    started = time.time()
    failed = []

    for name, fn in groups.items():
        print(f"\n[{name}]")
        try:
            fn(args.force)
        except Exception:  # noqa: BLE001
            failed.append(name)
            print(f"  ✗ {name} FAILED")
            traceback.print_exc()

    mins = (time.time() - started) / 60
    print(f"\n{'-' * 60}")
    if failed:
        print(f"FINISHED WITH FAILURES: {', '.join(failed)}  ({mins:.1f} min)")
        return 1
    print(f"All artifacts built in {mins:.1f} min → {artifacts.ART_DIR}")
    print("The app will now load Section 6 instantly. Every page still has a "
          "▶ Run button to train live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
