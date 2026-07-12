"""Shared mock datasets for Section 3, so pages tell one continuous story.

The running example: patients measured on two blood biomarkers (A and B,
both on a 0-10 scale), labelled healthy (0) or disease (1). Generated with
fixed seeds, so every page sees identical data.
"""

import numpy as np


def patients(n_per_class=60, seed=7, spread=1.25):
    """Two overlapping Gaussian blobs: (X, y) with X = [biomarker A, B]."""
    r = np.random.default_rng(seed)
    healthy = r.normal([3.6, 4.0], spread, (n_per_class, 2))
    disease = r.normal([6.4, 6.2], spread, (n_per_class, 2))
    X = np.vstack([healthy, disease]).clip(0, 10)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    order = r.permutation(len(y))
    return X[order], y[order]


def patients_split(seed=7):
    """The standard split used across Section 3: 80 train / 40 validation."""
    X, y = patients(seed=seed)
    return X[:80], X[80:], y[:80], y[80:]


def machines(n=1000, risk="linear", seed=0, lambda_max=5.0, r=0.5,
             d=10, censor_frac=0.5):
    """Time-to-failure data for industrial machines, generated exactly the way
    Katzman et al. (2018) generate their simulated survival data — but with
    machines instead of patients.

    Each machine has d sensor covariates ~ Uniform[-1, 1). Its true log-risk
    h(x) depends on only the first two:

      risk="linear"     h(x) = x0 + 2*x1                     (paper's linear)
      risk="nonlinear"  h(x) = log(lambda_max) *
                               exp(-(x0^2 + x1^2) / (2 r^2)) (paper's gaussian)

    Failure times are drawn from an exponential distribution whose rate is
    exp(h(x)) — i.e. a proportional-hazards generative model with a constant
    baseline hazard. Then ~censor_frac of machines are right-censored at a
    uniformly drawn inspection time before their failure.

    Returns (X, time, event) where event=1 means "we saw it fail" and
    event=0 means "still working when we stopped watching" (censored).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    if risk == "linear":
        h = X[:, 0] + 2 * X[:, 1]
    elif risk == "nonlinear":
        h = np.log(lambda_max) * np.exp(
            -(X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * r ** 2))
    else:
        raise ValueError("risk must be 'linear' or 'nonlinear'")
    # exponential survival time with rate exp(h): higher risk -> fails sooner
    true_time = rng.exponential(scale=np.exp(-h))
    event = np.ones(n, dtype=int)
    n_cens = int(censor_frac * n)
    cens_idx = rng.choice(n, n_cens, replace=False)
    # censor at a uniform time between 0 and the machine's true failure time
    obs_time = true_time.copy()
    obs_time[cens_idx] = rng.uniform(0, true_time[cens_idx])
    event[cens_idx] = 0
    return X, obs_time, event


def machines_treatment(n=1000, seed=0, lambda_max=10.0, r=0.5, d=10):
    """The paper's treatment-recommender simulation, as equipment servicing.

    Half the machines get servicing regime A (trt=0), half regime B (trt=1).
    Regime B only helps machines whose sensor profile sits in the gaussian
    high-risk region — so the *right* regime depends on the machine. A linear
    Cox model cannot express this; DeepSurv can.

    Returns (X, time, event, trt) with trt appended as the LAST column of X.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    trt = rng.integers(0, 2, n)
    gauss = np.log(lambda_max) * np.exp(
        -(X[:, 0] ** 2 + X[:, 1] ** 2) / (2 * r ** 2))
    # regime B (trt=1) removes the gaussian excess risk; regime A leaves it
    h = np.where(trt == 1, 0.0, gauss)
    true_time = rng.exponential(scale=np.exp(-h))
    event = np.ones(n, dtype=int)
    cens_idx = rng.choice(n, n // 2, replace=False)
    obs_time = true_time.copy()
    obs_time[cens_idx] = rng.uniform(0, true_time[cens_idx])
    event[cens_idx] = 0
    X_full = np.column_stack([X, trt])
    return X_full, obs_time, event, trt


def pumps(n_per_class=60, seed=17, spread=1.2):
    """Industrial pumps on a test rig: X = [vibration, temperature] (both
    scaled 0-10), y = 0 (passed inspection) / 1 (failed within a month)."""
    r = np.random.default_rng(seed)
    ok = r.normal([3.4, 3.8], spread, (n_per_class, 2))
    fail = r.normal([6.6, 6.4], spread, (n_per_class, 2))
    X = np.vstack([ok, fail]).clip(0, 10)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    order = r.permutation(len(y))
    return X[order], y[order]


def pumps_xor(n=240, seed=18):
    """The nasty batch: a pump fails when EXACTLY ONE of the two readings is
    high (high vibration OR high temperature, but not both — think two
    faults that cancel out). This is an XOR pattern: no straight line can
    separate it, which is precisely why hidden layers were invented."""
    r = np.random.default_rng(seed)
    X = r.uniform(0, 10, (n, 2))
    # keep a margin band around 5 so the classes don't kiss at the border
    keep = (np.abs(X[:, 0] - 5) > 0.7) & (np.abs(X[:, 1] - 5) > 0.7)
    X = X[keep]
    y = ((X[:, 0] > 5) != (X[:, 1] > 5)).astype(int)
    return X, y


def rings(n_per_class=70, seed=12):
    """A disease that strikes at BOTH extremes: healthy patients form a ring
    of moderate values around a diseased core (not linearly separable)."""
    r = np.random.default_rng(seed)
    angles = r.uniform(0, 2 * np.pi, n_per_class)
    radius = r.normal(3.2, 0.35, n_per_class)
    outer = np.column_stack([5 + radius * np.cos(angles),
                             5 + radius * np.sin(angles)])
    inner = r.normal([5, 5], 0.9, (n_per_class, 2))
    X = np.vstack([outer, inner]).clip(0, 10)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    order = r.permutation(len(y))
    return X[order], y[order]
