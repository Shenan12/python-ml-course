"""The heavy computations behind Section 7 (DeepHit), in one place.

Same pattern as Section 6: no Streamlit here. `build_artifacts.py` runs these
once and saves the results; the pages load them instantly and offer ▶ Run live
buttons. Every model is deliberately small (two 32-node layers, ~100 epochs on
15k rows) — this whole file runs in a couple of minutes on a laptop CPU.
"""

from __future__ import annotations

# MUST come before pycox/torchtuples: redirects their scratch files to temp so
# the app also runs on read-only deployments (Streamlit Cloud). See utils/compat.
from utils import compat

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchtuples as tt  # noqa: E402

compat.patch_torchtuples()  # EarlyStopping checkpoints -> temp, not repo dir

from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import proportional_hazard_test  # noqa: E402
from pycox.evaluation import EvalSurv  # noqa: E402
from pycox.models import CoxPH, DeepHitSingle  # noqa: E402

from utils import cricket  # noqa: E402

# The tuned DeepHit settings, found by the search on the "Discretising Time"
# and "The Two-Part Loss" pages. Small on purpose.
BEST = dict(scheme="quantiles", num_bins=20, alpha=0.5, sigma=0.5, lr=0.001)
LAYERS = [32, 32]
EPOCHS = 100
BATCH = 256


def _data():
    df = cricket.engineer(cricket.load_raw())
    cols = cricket.feature_columns(df)
    tr, va, te = cricket.splits(df)
    Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
    Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
    Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)
    return dict(df=df, cols=cols, tr=tr, va=va, te=te,
                Xtr=Xtr, dtr=dtr, etr=etr, Xva=Xva, dva=dva, eva=eva,
                Xte=Xte, dte=dte, ete=ete, mu=mu, sd=sd)


def _eval(surv_df, dte, ete):
    ev = EvalSurv(surv_df, dte, ete, censor_surv="km")
    grid = np.linspace(float(dte.min()), float(dte.max()), 50)
    return float(ev.concordance_td()), float(ev.integrated_brier_score(grid))


def fit_deephit(D, scheme=BEST["scheme"], num_bins=BEST["num_bins"],
                alpha=BEST["alpha"], sigma=BEST["sigma"], lr=BEST["lr"],
                seed=0):
    lab = DeepHitSingle.label_transform(num_bins, scheme=scheme)
    ytr = lab.fit_transform(D["dtr"], D["etr"])
    yva = lab.transform(D["dva"], D["eva"])
    torch.manual_seed(seed)
    net = tt.practical.MLPVanilla(len(D["cols"]), LAYERS, lab.out_features,
                                  batch_norm=True, dropout=0.1)
    model = DeepHitSingle(net, tt.optim.Adam(lr), alpha=alpha, sigma=sigma,
                          duration_index=lab.cuts)
    model.fit(D["Xtr"], ytr, batch_size=BATCH, epochs=EPOCHS, verbose=False,
              val_data=(D["Xva"], yva),
              callbacks=[tt.callbacks.EarlyStopping(patience=15)])
    return model, lab


def fit_deepsurv(D, seed=0):
    torch.manual_seed(seed)
    net = tt.practical.MLPVanilla(len(D["cols"]), LAYERS, 1, batch_norm=True,
                                  dropout=0.1, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(0.01))
    model.fit(D["Xtr"], (D["dtr"], D["etr"]), batch_size=BATCH, epochs=EPOCHS,
              verbose=False, val_data=(D["Xva"], (D["dva"], D["eva"])),
              callbacks=[tt.callbacks.EarlyStopping(patience=15)])
    model.compute_baseline_hazards()
    return model


# ------------------------------------------------------------------ h1
def ph_violation_experiment() -> dict:
    """Does proportional hazards actually hold for cricket? Test it."""
    df = cricket.engineer(cricket.load_raw())

    # the empirical dismissal hazard, ball by ball
    maxb = 100
    at_risk = np.array([(df.balls_faced >= b).sum() for b in range(1, maxb + 1)])
    outs = np.array([((df.balls_faced == b) & (df.event == 1)).sum()
                     for b in range(1, maxb + 1)])
    hazard = outs / np.maximum(at_risk, 1)

    # Kaplan-Meier by batting position group
    km = {}
    groups = {"openers (1-2)": df.batting_position <= 2,
              "middle order (3-5)": df.batting_position.between(3, 5),
              "lower order (6+)": df.batting_position >= 6}
    for name, mask in groups.items():
        k = KaplanMeierFitter().fit(df.loc[mask, "balls_faced"],
                                    df.loc[mask, "event"])
        sf = k.survival_function_.iloc[:, 0]
        km[f"km_{name}_t"] = sf.index.values.astype(float)
        km[f"km_{name}_s"] = sf.values.astype(float)

    # the formal test: Schoenfeld residuals
    feats = ["career_avg", "career_sr", "batting_position", "is_chasing",
             "wickets_at_entry"]
    d = df[feats + ["balls_faced", "event"]].dropna()
    cph = CoxPHFitter().fit(d, "balls_faced", "event")
    zph = proportional_hazard_test(cph, d, time_transform="rank")
    tests = zph.summary[["test_statistic", "p"]].reset_index()
    tests.columns = ["covariate", "test_statistic", "p"]

    # a time-varying hazard ratio: openers vs the rest, early vs late
    hr = []
    for lo, hi in [(1, 10), (11, 25), (26, 50), (51, 100)]:
        sub = df[(df.balls_faced >= lo)].copy()
        sub["window_event"] = ((sub.balls_faced <= hi) & (sub.event == 1)).astype(int)
        sub["window_dur"] = np.minimum(sub.balls_faced, hi)
        c = CoxPHFitter().fit(
            sub.assign(opener=(sub.batting_position <= 2).astype(int))
               [["opener", "window_dur", "window_event"]],
            "window_dur", "window_event")
        hr.append({"window": f"balls {lo}-{hi}",
                   "hazard_ratio_opener": float(np.exp(c.params_["opener"])),
                   "p": float(c.summary.loc["opener", "p"])})

    out = {
        "hazard": hazard, "balls": np.arange(1, maxb + 1).astype(float),
        "tests_csv": tests.to_csv(index=False),
        "hr_csv": pd.DataFrame(hr).to_csv(index=False),
        "cox_cindex": float(cph.concordance_index_),
        "n": int(len(df)), "n_events": int(df.event.sum()),
        "censor_rate": float(1 - df.event.mean()),
    }
    out.update(km)
    return out


# ------------------------------------------------------------------ h2
def leakage_experiment() -> dict:
    """Random split vs time-ordered split — how much does leakage flatter you?"""
    D = _data()
    df, cols = D["df"], D["cols"]

    # honest: time-ordered
    ds = fit_deepsurv(D)
    c_time, _ = _eval(ds.predict_surv_df(D["Xte"]), D["dte"], D["ete"])

    # dishonest: random split of the SAME rows
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(df))
    n_tr, n_va = len(D["tr"]), len(D["va"])
    r_tr = df.iloc[idx[:n_tr]]
    r_va = df.iloc[idx[n_tr:n_tr + n_va]]
    r_te = df.iloc[idx[n_tr + n_va:]]
    Xtr, dtr, etr, mu, sd = cricket.xy(r_tr, cols)
    Xva, dva, eva, _, _ = cricket.xy(r_va, cols, mu, sd)
    Xte, dte, ete, _, _ = cricket.xy(r_te, cols, mu, sd)
    D2 = dict(D, cols=cols, Xtr=Xtr, dtr=dtr, etr=etr, Xva=Xva, dva=dva,
              eva=eva, Xte=Xte, dte=dte, ete=ete)
    ds2 = fit_deepsurv(D2)
    c_random, _ = _eval(ds2.predict_surv_df(Xte), dte, ete)

    return {"c_time_split": c_time, "c_random_split": c_random,
            "n_train": int(len(D["tr"])), "n_val": int(len(D["va"])),
            "n_test": int(len(D["te"])),
            "train_years": f"{D['tr'].year.min()}–{D['tr'].year.max()}",
            "test_years": f"{D['te'].year.min()}–{D['te'].year.max()}",
            "n_features": int(len(cols)),
            "features_csv": pd.DataFrame({"feature": cols}).to_csv(index=False)}


# ------------------------------------------------------------------ h3
def binning_experiment() -> dict:
    """The discretisation trap: equidistant vs quantile bins, and how many."""
    D = _data()
    rows = []
    for scheme in ["equidistant", "quantiles"]:
        for nb in [5, 10, 20, 30]:
            model, lab = fit_deephit(D, scheme=scheme, num_bins=nb)
            c, ibs = _eval(model.predict_surv_df(D["Xte"]), D["dte"], D["ete"])
            rows.append({"scheme": scheme, "num_bins": nb,
                         "C-td": c, "IBS": ibs,
                         "cuts": np.round(lab.cuts, 1).tolist()})
    # the bin edges each scheme chooses, for the visualization
    cuts = {}
    for scheme in ["equidistant", "quantiles"]:
        lab = DeepHitSingle.label_transform(20, scheme=scheme)
        lab.fit(D["dtr"], D["etr"])
        cuts[f"cuts_{scheme}"] = np.asarray(lab.cuts, dtype=float)
    out = {"results_csv": pd.DataFrame(rows).to_csv(index=False),
           "durations": D["dtr"].astype(float)}
    out.update(cuts)
    return out


# ------------------------------------------------------------------ h4/h5
def alpha_experiment() -> dict:
    """The two-part loss: sweep alpha (likelihood vs ranking weight)."""
    D = _data()
    rows = []
    for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
        model, _ = fit_deephit(D, alpha=alpha)
        c, ibs = _eval(model.predict_surv_df(D["Xte"]), D["dte"], D["ete"])
        rows.append({"alpha": alpha, "C-td": c, "IBS": ibs})
    return {"results_csv": pd.DataFrame(rows).to_csv(index=False)}


def pmf_experiment() -> dict:
    """The tuned DeepHit's PMF / survival curves for a few real batters."""
    D = _data()
    model, lab = fit_deephit(D)
    # NOTE the orientation: predict_pmf returns (n_samples, n_bins), so one
    # batter's PMF is a ROW. Getting this wrong silently produces a "PMF"
    # sliced across people instead of across time.
    pmf = model.predict_pmf(D["Xte"])          # (n, bins)
    surv = model.predict_surv_df(D["Xte"])     # (bins, n)

    te = D["te"].reset_index(drop=True)
    # pick a settled opener, a middle-order batter, and a tailender
    picks = {}
    for label, mask in [
        ("top-order, in form", (te.batting_position <= 2) & (te.career_avg > 40)),
        ("middle order", te.batting_position.between(4, 5)),
        ("tailender", te.batting_position >= 8),
    ]:
        idx = int(np.flatnonzero(mask.to_numpy())[0]) if mask.any() else 0
        picks[label] = idx

    out = {"cuts": np.asarray(lab.cuts, dtype=float),
           "surv_index": surv.index.values.astype(float)}
    meta = []
    for label, i in picks.items():
        key = label.replace(" ", "_").replace(",", "")
        out[f"pmf_{key}"] = np.asarray(pmf[i, :], dtype=float)
        out[f"surv_{key}"] = np.asarray(surv.iloc[:, i].values, dtype=float)
        row = te.iloc[i]
        meta.append({"label": label, "batter": row["batter"],
                     "position": int(row["batting_position"]),
                     "career_avg": float(row["career_avg"]),
                     "balls_faced": int(row["balls_faced"]),
                     "event": int(row["event"])})
    out["meta_csv"] = pd.DataFrame(meta).to_csv(index=False)
    return out


# ------------------------------------------------------------------ h7
def head_to_head_experiment() -> dict:
    """DeepSurv vs DeepHitSingle on the same cricket data, honestly."""
    D = _data()

    cox = CoxPHFitter(penalizer=0.01)
    tr_df = pd.DataFrame(D["Xtr"], columns=D["cols"])
    tr_df["dur"], tr_df["ev"] = D["dtr"], D["etr"]
    cox.fit(tr_df, "dur", "ev")
    cox_risk = D["Xte"] @ cox.params_.values
    from lifelines.utils import concordance_index
    c_cox = float(concordance_index(D["dte"], -cox_risk, D["ete"]))

    ds = fit_deepsurv(D)
    surv_ds = ds.predict_surv_df(D["Xte"])
    c_ds, ibs_ds = _eval(surv_ds, D["dte"], D["ete"])

    dh, lab = fit_deephit(D)
    surv_dh = dh.predict_surv_df(D["Xte"])
    c_dh, ibs_dh = _eval(surv_dh, D["dte"], D["ete"])

    # mean predicted survival curve of each model vs the Kaplan-Meier truth
    km = KaplanMeierFitter().fit(D["dte"], D["ete"])
    km_t = km.survival_function_.index.values.astype(float)
    km_s = km.survival_function_.iloc[:, 0].values.astype(float)

    return {
        "c_cox": c_cox, "c_ds": c_ds, "c_dh": c_dh,
        "ibs_ds": ibs_ds, "ibs_dh": ibs_dh,
        "ds_index": surv_ds.index.values.astype(float),
        "ds_mean": surv_ds.mean(axis=1).values.astype(float),
        "dh_index": surv_dh.index.values.astype(float),
        "dh_mean": surv_dh.mean(axis=1).values.astype(float),
        "km_t": km_t, "km_s": km_s,
        "n_test": int(len(D["dte"])),
    }


# ================================================================
SNIPPETS: dict[str, str] = {}

SNIPPETS["h2_features"] = '''import pandas as pd
from utils import cricket

df = cricket.engineer(cricket.load_raw())

print("one row = one batter's innings")
print(df[["batter", "balls_faced", "runs", "event", "batting_position",
          "career_avg", "wickets_at_entry"]].head(8).to_string(index=False))

print()
print("event = 1 -> dismissed;  event = 0 -> NOT OUT (censored)")
print(f"  dismissed: {df.event.sum():,}")
print(f"  not out  : {(1 - df.event).sum():,} "
      f"({1 - df.event.mean():.1%} of innings)")
print()

# the time-ordered split: train on the past, test on the future
train, val, test = cricket.splits(df)
print(f"train {len(train):,} innings ({train.year.min()}-{train.year.max()})")
print(f"val   {len(val):,} innings ({val.year.min()}-{val.year.max()})")
print(f"test  {len(test):,} innings ({test.year.min()}-{test.year.max()})")
print()
print(f"{len(cricket.feature_columns(df))} features:")
print(" ", cricket.feature_columns(df))'''

SNIPPETS["h3_binning"] = '''import numpy as np
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import scipy.integrate as si
if not hasattr(si, "simps"):
    si.simps = si.simpson          # pycox expects an old SciPy name

from utils import cricket

df = cricket.engineer(cricket.load_raw())
cols = cricket.feature_columns(df)
tr, va, te = cricket.splits(df)
Xtr, dtr, etr, mu, sd = cricket.xy(tr, cols)
Xva, dva, eva, _, _ = cricket.xy(va, cols, mu, sd)
Xte, dte, ete, _, _ = cricket.xy(te, cols, mu, sd)

for scheme in ["equidistant", "quantiles"]:
    # DISCRETISE: chop "balls faced" into 20 bins
    labtrans = DeepHitSingle.label_transform(20, scheme=scheme)
    y_train = labtrans.fit_transform(dtr, etr)
    y_val = labtrans.transform(dva, eva)

    torch.manual_seed(0)
    net = tt.practical.MLPVanilla(len(cols), [32, 32], labtrans.out_features,
                                  batch_norm=True, dropout=0.1)
    model = DeepHitSingle(net, tt.optim.Adam(0.001), alpha=0.5, sigma=0.5,
                          duration_index=labtrans.cuts)
    model.fit(Xtr, y_train, batch_size=256, epochs=100, verbose=False,
              val_data=(Xva, y_val),
              callbacks=[tt.callbacks.EarlyStopping(patience=15)])

    ev = EvalSurv(model.predict_surv_df(Xte), dte, ete, censor_surv="km")
    print(f"{scheme:12}: C-td {ev.concordance_td():.4f}   "
          f"first 6 bin edges: {np.round(labtrans.cuts[:6], 1)}")'''
