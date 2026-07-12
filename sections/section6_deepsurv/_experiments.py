"""The heavy computations behind Section 6, in one place.

Nothing in this module touches Streamlit. That matters, because these exact
functions are called from two places:

  * `build_artifacts.py`, which runs them once and saves the results, and
  * the pages themselves, when you press a ▶ Run live button.

Keeping them here guarantees the numbers the pages *show* and the numbers the
build script *computes* come from identical code.

SNIPPETS holds the code strings for the teaching examples that are expensive to
run (they train real networks). Pages display them and show their saved output;
the build script executes them to produce that output. Because both sides read
the same string, the code on screen and the output below it can never drift
apart — and `utils.artifacts.code_hash` enforces that with a checksum.
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
from lifelines.utils import concordance_index  # noqa: E402
from pycox.models import CoxPH  # noqa: E402

from utils.mockdata import machines, machines_treatment  # noqa: E402

# The slider options the pages offer. build_artifacts.py precomputes every
# combination of these, so every slider position loads instantly.
D7_RISKS = ["linear", "nonlinear"]
D7_TRAIN_SIZES = [500, 1000, 2000, 4000]
# Early stopping is a toggle rather than an epoch count, because with early
# stopping ON the run halts long before any of the epoch caps we might offer —
# so an "epochs" slider would have been inert (it changed nothing). Turning it
# OFF trains the full 512 epochs and lets you watch overfitting instead.
D7_EARLY_STOP = [True, False]
D7_MAX_EPOCHS = 512
D8_MAX_TRIALS = 20
SENSOR_SCALES = np.array([1, 500, 0.01, 50, 1, 200, 0.05, 1, 10, 1000.0])


def _standardizer(X):
    mu, sd = X.mean(0), X.std(0)
    return mu, sd, lambda A: ((A - mu) / sd).astype("float32")


def _cox_nll(h, T, E):
    order = torch.argsort(T, descending=True)
    h, E = h[order], E[order]
    return -((h - torch.logcumsumexp(h, 0)) * E).sum() / E.sum()


def _deepsurv(in_features=10, nodes=32, depth=2, dropout=0.1):
    return tt.practical.MLPVanilla(in_features, [nodes] * depth, 1,
                                   batch_norm=True, dropout=dropout,
                                   output_bias=False)


# --------------------------------------------------------------- d5
def standardization_experiment(seed: int = 0) -> dict:
    """Does standardizing the inputs actually matter? Test all four cells of
    {raw, standardized} × {plain SGD, Adam}.

    The answer turned out to be more interesting than "yes": Adam's
    per-parameter adaptive step sizes largely compensate for badly-scaled
    inputs, whereas plain SGD on raw sensors diverges to NaN outright. So
    standardization is best understood as *insurance against divergence*, not
    as a guaranteed accuracy win. The page reports whatever this returns.
    """
    X, T, E = machines(n=800, risk="nonlinear", seed=1)
    Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)
    Xr, Xter = X * SENSOR_SCALES, Xte * SENSOR_SCALES
    scaled = ((Xr - Xr.mean(0)) / Xr.std(0), (Xter - Xr.mean(0)) / Xr.std(0))

    out = {}
    for scaling, (Xa, Xb) in [("raw", (Xr, Xter)),
                              ("standardized", scaled)]:
        for optname in ["SGD", "Adam"]:
            torch.manual_seed(seed)
            net = torch.nn.Sequential(
                torch.nn.Linear(10, 32), torch.nn.ReLU(),
                torch.nn.Dropout(0.1), torch.nn.Linear(32, 1, bias=False))
            opt = (torch.optim.SGD(net.parameters(), lr=0.01)
                   if optname == "SGD"
                   else torch.optim.Adam(net.parameters(), lr=0.01))
            Xt = torch.tensor(np.asarray(Xa), dtype=torch.float32)
            Tt = torch.tensor(T, dtype=torch.float32)
            Et = torch.tensor(E, dtype=torch.float32)
            hist = []
            for _ in range(200):
                net.train()
                opt.zero_grad()
                loss = _cox_nll(net(Xt).squeeze(1), Tt, Et)
                loss.backward()
                opt.step()
                hist.append(float(loss.item()))
            net.eval()
            with torch.no_grad():
                r = net(torch.tensor(np.asarray(Xb),
                                     dtype=torch.float32)).squeeze(1).numpy()
            diverged = not np.all(np.isfinite(r))
            key = f"{scaling}_{optname}"
            out[f"{key}_hist"] = np.asarray(hist, dtype=float)
            out[f"{key}_diverged"] = bool(diverged)
            out[f"{key}_cindex"] = (
                float("nan") if diverged
                else float(concordance_index(Tte, -r, Ete)))
    return out


# --------------------------------------------------------------- d6
def recommender_experiment(seed: int = 0) -> dict:
    """Fit Cox and DeepSurv on the treatment simulation.

    Returns plain arrays only (no torch model), so the result is picklable and
    can be cached to disk.
    """
    X, T, E, trt = machines_treatment(n=1000, seed=4)
    cols = [f"x{i}" for i in range(10)] + ["trt"]
    df = pd.DataFrame(X, columns=cols)
    df["T"], df["E"] = T, E
    cph = CoxPHFitter().fit(df, "T", "E")
    gamma = float(cph.params_["trt"])

    mu, sd, f = _standardizer(X)
    torch.manual_seed(seed)
    net = _deepsurv(in_features=11)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    Xs = torch.tensor(f(X))
    Tt = torch.tensor(T, dtype=torch.float32)
    Et = torch.tensor(E, dtype=torch.float32)
    for _ in range(400):
        net.train()
        opt.zero_grad()
        loss = _cox_nll(net(Xs).squeeze(1), Tt, Et)
        loss.backward()
        opt.step()
    net.eval()

    def rec(rows):
        g1, g0 = rows.copy(), rows.copy()
        g1[:, 10], g0[:, 10] = 1.0, 0.0
        with torch.no_grad():
            h1 = net(torch.tensor(f(g1))).squeeze(1).numpy()
            h0 = net(torch.tensor(f(g0))).squeeze(1).numpy()
        return h1 - h0

    # the recommendation surface over sensors 0 and 1
    g1, g2 = np.meshgrid(np.linspace(-1, 1, 60), np.linspace(-1, 1, 60))
    grid = np.zeros((g1.size, 11))
    grid[:, 0], grid[:, 1] = g1.ravel(), g2.ravel()
    rec_surface = rec(grid).reshape(g1.shape)

    rec_actual = rec(X.copy())
    recommended = (rec_actual < 0).astype(int)
    agreed = recommended == trt

    med_agree = float(KaplanMeierFitter().fit(
        T[agreed], E[agreed]).median_survival_time_)
    med_disagree = float(KaplanMeierFitter().fit(
        T[~agreed], E[~agreed]).median_survival_time_)

    return {
        "gamma": gamma,
        "rec_surface": rec_surface,
        "rec_actual": rec_actual,
        "agreed": agreed.astype(int),
        "T": T, "E": E, "trt": trt, "X": X,
        "median_agree": med_agree,
        "median_disagree": med_disagree,
        "n_agree": int(agreed.sum()),
        "n_disagree": int((~agreed).sum()),
        "sample_rows": np.array([3, 17, 42, 88, 150]),
        "sample_recs": rec(X[[3, 17, 42, 88, 150]].copy()),
    }


# --------------------------------------------------------------- d7
def simulation_experiment(risk: str, n_train: int, early_stop: bool = True,
                          seed: int = 0) -> dict:
    """The paper's linear-vs-nonlinear experiment: Cox vs DeepSurv.

    `early_stop` halts training when validation loss stops improving (and
    restores the best weights). With it off, the net trains the full
    D7_MAX_EPOCHS and you can watch it overfit.
    """
    Xtr, Ttr, Etr = machines(n=n_train, risk=risk, seed=1)
    Xva, Tva, Eva = machines(n=1000, risk=risk, seed=3)
    Xte, Tte, Ete = machines(n=1000, risk=risk, seed=2)
    mu, sd, f = _standardizer(Xtr)

    cols = [f"x{i}" for i in range(10)]
    df = pd.DataFrame(f(Xtr), columns=cols)
    df["T"], df["E"] = Ttr, Etr
    cph = CoxPHFitter(penalizer=0.01).fit(df, "T", "E")
    cox_risk = f(Xte) @ cph.params_.values
    c_cox = float(concordance_index(Tte, -cox_risk, Ete))

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CoxPH(_deepsurv(), tt.optim.Adam(0.01))
    callbacks = ([tt.callbacks.EarlyStopping(patience=30)] if early_stop
                 else [])
    log = model.fit(
        f(Xtr), (Ttr.astype("float32"), Etr.astype("float32")),
        batch_size=256, epochs=D7_MAX_EPOCHS, verbose=False,
        val_data=(f(Xva), (Tva.astype("float32"), Eva.astype("float32"))),
        callbacks=callbacks)
    ds_risk = model.predict(f(Xte)).ravel()
    c_ds = float(concordance_index(Tte, -ds_risk, Ete))

    history = log.to_pandas()
    train_curve = np.asarray(history["train_loss"].values, dtype=float)
    val_curve = np.asarray(history["val_loss"].values, dtype=float)
    epochs_run = int(len(train_curve))

    g1, g2 = np.meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 120))
    grid = np.zeros((120 * 120, 10))
    grid[:, 0], grid[:, 1] = g1.ravel(), g2.ravel()
    cox_surface = (f(grid) @ cph.params_.values).reshape(120, 120)
    ds_surface = model.predict(f(grid)).ravel().reshape(120, 120)

    return {
        "c_cox": c_cox, "c_ds": c_ds,
        "cox_surface": cox_surface, "ds_surface": ds_surface,
        "betas": cph.params_.values.astype(float),
        "pvalues": cph.summary["p"].values.astype(float),
        "cox_risk": np.asarray(cox_risk, dtype=float),
        "ds_risk": np.asarray(ds_risk, dtype=float),
        "Tte": Tte, "Ete": Ete,
        "train_curve": train_curve, "val_curve": val_curve,
        "epochs_run": epochs_run, "early_stop": bool(early_stop),
        "max_epochs": int(D7_MAX_EPOCHS),
    }


# --------------------------------------------------------------- d8
def random_search_experiment(n_trials: int = D8_MAX_TRIALS,
                             seed: int = 7) -> dict:
    """The paper's random hyperparameter search, rebuilt.

    Trials are drawn from a single sequential RNG, so the first k trials of an
    n-trial run are *exactly* the trials a k-trial run would produce. That is
    why the pages can serve the 5- and 10-trial views from this one 20-trial
    artifact without cheating.
    """
    # same data regime as the d7 simulation page, so the two pages' C-indices
    # are directly comparable (a tuned model here should reach d7's ballpark)
    Xtr, Ttr, Etr = machines(n=4000, risk="nonlinear", seed=1)
    Xva, Tva, Eva = machines(n=1000, risk="nonlinear", seed=3)
    Xte, Tte, Ete = machines(n=1000, risk="nonlinear", seed=2)
    _, _, f = _standardizer(Xtr)
    ytr = (Ttr.astype("float32"), Etr.astype("float32"))
    yva = (Tva.astype("float32"), Eva.astype("float32"))

    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trials):
        depth = int(rng.integers(1, 4))
        nodes = int(rng.choice([8, 16, 32, 48]))
        lr = float(10 ** rng.uniform(-3.5, -1.0))
        dropout = float(rng.uniform(0.1, 0.6))
        wd = float(10 ** rng.uniform(-5, -2))

        torch.manual_seed(0)
        model = CoxPH(_deepsurv(nodes=nodes, depth=depth, dropout=dropout),
                      tt.optim.Adam(lr, weight_decay=wd))
        model.fit(f(Xtr), ytr, batch_size=256, epochs=300, verbose=False,
                  val_data=(f(Xva), yva),
                  callbacks=[tt.callbacks.EarlyStopping(patience=30)])
        rows.append({
            "trial": t + 1, "depth": depth, "nodes": nodes,
            "learning rate": lr, "dropout": dropout,
            "L2 (weight decay)": wd,
            "validation C-index": float(
                concordance_index(Tva, -model.predict(f(Xva)).ravel(), Eva)),
            "test C-index": float(
                concordance_index(Tte, -model.predict(f(Xte)).ravel(), Ete)),
        })
    df = pd.DataFrame(rows)
    return {"trials_csv": df.to_csv(index=False)}


def trials_frame(payload: dict, n_trials: int) -> pd.DataFrame:
    """The first `n_trials` rows of the stored search — exactly what a run of
    that budget produces (see `random_search_experiment`)."""
    import io
    df = pd.read_csv(io.StringIO(payload["trials_csv"]))
    return df.head(n_trials).reset_index(drop=True)


# ================================================================
# The expensive teaching snippets. Shown on the pages; executed by
# build_artifacts.py to produce the output the pages display.
# ================================================================

SNIPPETS: dict[str, str] = {}

SNIPPETS["d4_example"] = '''import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from utils.mockdata import machines

# 800 machines, 10 sensors, NONLINEAR true risk (the paper's gaussian)
X, T, E = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)

mu, sd = X.mean(0), X.std(0)                 # standardize (paper does this)
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Xts = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
Tt = torch.tensor(T, dtype=torch.float32)
Et = torch.tensor(E, dtype=torch.float32)

torch.manual_seed(0)
net = nn.Sequential(                          # DeepSurv, exactly as described
    nn.Linear(10, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
    nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.1),
    nn.Linear(32, 1, bias=False),
)

def cox_nll(h, T, E):
    order = torch.argsort(T, descending=True)  # sort so risk sets are cumulative
    h, E = h[order], E[order]
    log_denom = torch.logcumsumexp(h, dim=0)   # every denominator in one pass
    return -((h - log_denom) * E).sum() / E.sum()

opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)  # L2!
for epoch in range(300):
    net.train()
    opt.zero_grad()
    loss = cox_nll(net(Xs).squeeze(1), Tt, Et)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}: loss {loss.item():.4f}")

net.eval()
with torch.no_grad():
    risk = net(Xts).squeeze(1).numpy()
print("test C-index:", round(concordance_index(Tte, -risk, Ete), 4))'''

SNIPPETS["d5_example"] = '''import numpy as np
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from utils.mockdata import machines

X, T, E = machines(n=800, risk="nonlinear", seed=1)
Xte, Tte, Ete = machines(n=400, risk="nonlinear", seed=2)

mu, sd = X.mean(0), X.std(0)                       # 1. STANDARDIZE
Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
Xts = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
Tt = torch.tensor(T, dtype=torch.float32)
Et = torch.tensor(E, dtype=torch.float32)

torch.manual_seed(0)
net = nn.Sequential(                               # 2. SELU (self-normalising)
    nn.Linear(10, 32), nn.SELU(), nn.AlphaDropout(0.1),
    nn.Linear(32, 32), nn.SELU(), nn.AlphaDropout(0.1),
    nn.Linear(32, 1, bias=False),
)

def cox_nll(h, T, E):
    order = torch.argsort(T, descending=True)
    h, E = h[order], E[order]
    return -((h - torch.logcumsumexp(h, 0)) * E).sum() / E.sum()

LR0, DECAY = 0.01, 0.001
opt = torch.optim.Adam(net.parameters(), lr=LR0, weight_decay=1e-4)  # 3. L2

for epoch in range(300):
    for g in opt.param_groups:                     # 4. LR DECAY
        g["lr"] = LR0 / (1 + epoch * DECAY)
    net.train()
    opt.zero_grad()
    loss = cox_nll(net(Xs).squeeze(1), Tt, Et)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # 5. CLIPPING
    opt.step()
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}: loss {loss.item():.4f}  "
              f"lr {opt.param_groups[0]['lr']:.5f}")

net.eval()
with torch.no_grad():
    risk = net(Xts).squeeze(1).numpy()
print("test C-index:", round(concordance_index(Tte, -risk, Ete), 4))'''

SNIPPETS["d7_example"] = '''import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

for risk in ["linear", "nonlinear"]:
    Xtr, Ttr, Etr = machines(n=2000, risk=risk, seed=1)
    Xte, Tte, Ete = machines(n=1000, risk=risk, seed=2)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    f = lambda A: ((A - mu) / sd).astype("float32")

    # --- linear Cox ---
    df = pd.DataFrame(f(Xtr), columns=[f"x{i}" for i in range(10)])
    df["T"], df["E"] = Ttr, Etr
    cph = CoxPHFitter(penalizer=0.01).fit(df, "T", "E")
    c_cox = concordance_index(Tte, -(f(Xte) @ cph.params_.values), Ete)

    # --- DeepSurv ---
    torch.manual_seed(0)
    net = tt.practical.MLPVanilla(10, [32, 32], 1, batch_norm=True,
                                  dropout=0.1, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(0.01))
    model.fit(f(Xtr), (Ttr.astype("float32"), Etr.astype("float32")),
              batch_size=256, epochs=300, verbose=False)
    c_ds = concordance_index(Tte, -model.predict(f(Xte)).ravel(), Ete)

    print(f"{risk:10}: Cox {c_cox:.3f} | DeepSurv {c_ds:.3f} | "
          f"beta_0={cph.params_['x0']:+.3f} beta_1={cph.params_['x1']:+.3f}")'''

SNIPPETS["d8_example"] = '''import numpy as np
import torch
import torchtuples as tt
from lifelines.utils import concordance_index
from pycox.models import CoxPH
from utils.mockdata import machines

Xtr, Ttr, Etr = machines(n=1500, risk="nonlinear", seed=1)
Xva, Tva, Eva = machines(n=700, risk="nonlinear", seed=3)
mu, sd = Xtr.mean(0), Xtr.std(0)
f = lambda A: ((A - mu) / sd).astype("float32")
ytr = (Ttr.astype("float32"), Etr.astype("float32"))
yva = (Tva.astype("float32"), Eva.astype("float32"))

rng = np.random.default_rng(7)
best = None
for trial in range(8):
    # RANDOM search: every hyperparameter gets a fresh value each trial
    depth   = int(rng.integers(1, 4))
    nodes   = int(rng.choice([8, 16, 32, 48]))
    lr      = float(10 ** rng.uniform(-3.5, -1.0))   # log-uniform!
    dropout = float(rng.uniform(0.1, 0.6))
    wd      = float(10 ** rng.uniform(-5, -2))

    torch.manual_seed(0)
    net = tt.practical.MLPVanilla(10, [nodes] * depth, 1, batch_norm=True,
                                  dropout=dropout, output_bias=False)
    model = CoxPH(net, tt.optim.Adam(lr, weight_decay=wd))
    model.fit(f(Xtr), ytr, batch_size=256, epochs=200, verbose=False,
              val_data=(f(Xva), yva),
              callbacks=[tt.callbacks.EarlyStopping(patience=20)])

    c_val = concordance_index(Tva, -model.predict(f(Xva)).ravel(), Eva)
    print(f"trial {trial+1}: depth={depth} nodes={nodes} "
          f"lr={lr:.4f} drop={dropout:.2f} -> val C-index {c_val:.4f}")
    if best is None or c_val > best[0]:
        best = (c_val, depth, nodes, lr, dropout, wd)

print(f"\\nWINNER: val C-index {best[0]:.4f} with depth={best[1]}, "
      f"nodes={best[2]}, lr={best[3]:.4f}, dropout={best[4]:.2f}")'''
