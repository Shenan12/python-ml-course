import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter

from utils.sandbox import guided_sandbox, show_example

st.title("⏳ Survival Analysis from Scratch")
st.markdown(
    """
Everything you've modelled so far predicted *a number* (regression) or *a
class* (classification). Survival analysis predicts **how long until an event
happens** — and it exists as a separate discipline because of one awkward
fact that breaks ordinary regression: **censoring**.

Our running example for this whole section: **industrial machines on a test
rig**, and we want to know how long each runs before it **fails**. (The
DeepSurv paper studies patients; the mathematics is identical, and we'll meet
its real medical datasets when we review the paper's results.)

Here's the problem. We watched 6 machines for 12 months:
"""
)

st.markdown(
    """
| machine | what we observed | time T | event E |
|---|---|---|---|
| M1 | failed at month 4 | 4 | **1** (failure observed) |
| M2 | still running when the study ended at month 12 | 12 | **0** (censored) |
| M3 | failed at month 7 | 7 | **1** |
| M4 | withdrawn at month 5 — sold to another plant, still working | 5 | **0** (censored) |
| M5 | failed at month 11 | 11 | **1** |
| M6 | still running at month 12 | 12 | **0** (censored) |

Machines M2, M4 and M6 are **right-censored**: we know their true lifetime is
*longer than* the time recorded, but not by how much. This is partial
information, and it is everywhere in real data (patients still alive at the
end of a trial, customers who haven't churned yet, batters not out at the end
of an innings).
"""
)

st.header("1 · Why you can't just use the tools you already have")
st.markdown(
    """
Three tempting shortcuts, and why each is wrong — this is the argument that
justifies the entire field:

| Shortcut | What breaks |
|---|---|
| **Drop the censored rows** | You throw away the machines that lasted *longest*. Your remaining sample is biased toward early failures, so every lifetime estimate comes out too short. In the paper's SUPPORT dataset that would mean discarding **68%** of the data. |
| **Treat T as the true lifetime and run linear regression** | You'd be telling the model "M2 failed at month 12" when M2 *didn't fail at all*. You are training on labels you know to be false. |
| **Turn it into classification: "failed within 12 months, yes/no"** | You bin away the timing (a failure at month 1 and month 11 become the same label), and you still can't classify M4, which left at month 5 — was it going to fail by month 12 or not? Unknowable. |

Survival analysis is the machinery for using **every** machine — censored or
not — for exactly what it does tell you. M4 contributes the honest statement
*"this machine survived at least 5 months"*, and that statement genuinely
constrains the model.
"""
)

st.header("2 · The two functions that define everything")
st.markdown(
    r"""
As a statistician you'll recognise these as re-expressions of a distribution
you already know. Let $T$ be the random lifetime, with density $f(t)$ and CDF
$F(t) = P(T \le t)$.

**The survival function** — just the complement of the CDF:

$$S(t) = P(T > t) = 1 - F(t)$$

"the probability a machine is still running at time $t$." It starts at 1 and
decreases.

**The hazard function** — the new idea, and the one everything hinges on:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t} = \frac{f(t)}{S(t)}$$

"given the machine has survived to time $t$, the *instantaneous rate* of
failing right now." It's a **conditional** rate, not a probability — it can
exceed 1. This conditioning is the whole trick: it's precisely what lets
censored machines contribute (they're in the "survived to $t$" denominator
right up until they leave).

The two are locked together by $S(t) = \exp\left(-\int_0^t h(u)\,du\right)$.
Know one, know the other. Play with the hazard below and watch the survival
curve respond:
"""
)

with st.expander("🎓 Deeper statistics — the likelihood of censored data "
                 "(the foundation for every loss in Sections 6 and 7)"):
    st.markdown(
        r"""
How does a censored observation *formally* enter a model? Through the
likelihood. For machine $i$ with recorded time $T_i$ and event flag
$E_i$:

- if it **failed** ($E_i = 1$), it contributes the density: $f(T_i)$ — "the lifetime was exactly $T_i$";
- if it was **censored** ($E_i = 0$), it contributes the survival function: $S(T_i)$ — "the lifetime exceeds $T_i$", which is *all we know*.

So the likelihood of the whole sample factorises as

$$L = \prod_{i=1}^{n} f(T_i)^{E_i}\, S(T_i)^{1-E_i} = \prod_{i=1}^{n} h(T_i)^{E_i}\, S(T_i)$$

(the second form uses $f = h \cdot S$ — hazard times survival). This one
line is the entire field in miniature: **maximum likelihood, where censored
units contribute probability-of-surviving instead of density-of-failing.**
Every loss function you will meet from here on is a version of it — the Cox
*partial* likelihood eliminates $h_0$ from it, DeepSurv's loss is that
partial likelihood with a network inside, and DeepHit's $\mathcal{L}_1$ is
its exact discrete-time analogue.

**The assumption that makes this valid:** censoring must be
**non-informative** (independent censoring) — the *reason* a machine leaves
the study must carry no information about its residual lifetime.
M4 was sold, which is fine *if* buyers don't systematically pick
soon-to-fail machines. If censoring is informative (e.g. engineers withdraw
machines that sound like they're about to fail), the factorisation above is
simply the wrong likelihood and **every** method on these pages — KM, Cox,
DeepSurv, DeepHit — inherits the bias. No amount of neural network fixes a
wrong likelihood.
"""
    )

shape = st.radio(
    "hazard shape",
    ["constant (memoryless — pure random failure)",
     "increasing (wear-out — the usual for machinery)",
     "decreasing (infant mortality — early defects burn off)",
     "bathtub (both: defects early, wear-out late)"],
    index=1,
)
t = np.linspace(0.01, 12, 400)
if shape.startswith("constant"):
    h = np.full_like(t, 0.15)
elif shape.startswith("increasing"):
    h = 0.02 * t ** 1.6
elif shape.startswith("decreasing"):
    h = 0.6 / (1 + 1.2 * t)
else:
    h = 0.45 / (1 + 3 * t) + 0.004 * t ** 2.2
S = np.exp(-np.cumsum(h) * (t[1] - t[0]))       # S(t) = exp(-∫h)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(t, h, color="#c62828", linewidth=2.4)
axes[0].set_xlabel("months")
axes[0].set_ylabel("hazard h(t)")
axes[0].set_title("hazard: rate of failing right now,\ngiven still alive",
                  fontsize=10)
axes[0].grid(alpha=0.25)
axes[1].plot(t, S, color="#1565c0", linewidth=2.4)
axes[1].set_xlabel("months")
axes[1].set_ylabel("S(t)")
axes[1].set_ylim(0, 1.02)
axes[1].set_title("survival: P(still running past t)\n"
                  "S(t) = exp(−∫h)", fontsize=10)
axes[1].grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
median_idx = int(np.argmin(np.abs(S - 0.5)))
st.info(
    f"With this hazard, half the machines have failed by month "
    f"**{t[median_idx]:.1f}** (where S crosses 0.5 — the **median survival "
    "time**, which is how survival results are usually reported, because the "
    "*mean* is often undefined when some units never fail)."
)

st.header("3 · Kaplan–Meier: estimating S(t) from censored data")
st.markdown(
    r"""
Given our 6 machines, how do we estimate $S(t)$ *without* assuming any
distribution? The **Kaplan–Meier estimator** — a non-parametric MLE, and one
of the most-cited results in all of statistics:

$$\hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)$$

At each observed **failure time** $t_i$: $d_i$ = how many failed then, $n_i$
= how many were **still at risk** (still being watched) just before then.
Multiply the survival fractions together — a chain of conditional
probabilities, exactly like $P(A \cap B) = P(A)P(B|A)$.

**Censored machines do their work in the denominator $n_i$**: they count as
at-risk right up to the moment they leave, then quietly drop out without ever
being counted as a failure. That's how KM uses partial information honestly.
Below, the estimator is computed on our 6 machines by hand *and* by
`lifelines`, live:
"""
)

with st.expander("🎓 Deeper statistics — why KM is *the* nonparametric MLE, "
                 "and Greenwood's variance"):
    st.markdown(
        r"""
"Non-parametric MLE" is a real theorem, not a slogan. Allow the distribution
of $T$ to be *anything* (mass allowed at every observed time), write the
censored-data likelihood from the expander above, and maximise. The maximiser
puts hazard mass only at observed failure times, with the discrete hazard at
time $t_i$ estimated by exactly what you'd guess:

$$\hat h_i = \frac{d_i}{n_i} \quad\text{— failures over at-risk, a binomial proportion.}$$

The KM product $\hat S(t) = \prod_{t_i \le t}(1 - \hat h_i)$ is then just the
discrete version of $S = \exp(-\int h)$: survive each failure time in turn.
So KM is what maximum likelihood gives you when you refuse to assume a
distribution — the same estimator-from-principle logic as the sample mean
being the MLE of a normal mean.

Because each $\hat h_i$ is a binomial proportion, the delta method gives the
classic **Greenwood formula** for the variance of the whole curve:

$$\widehat{\mathrm{Var}}\big(\hat S(t)\big) = \hat S(t)^2 \sum_{t_i \le t} \frac{d_i}{n_i (n_i - d_i)}$$

— which is where the confidence band `lifelines` draws around a KM curve
comes from. Note what drives it: the band widens **to the right**, because
late in the study $n_i$ has been eaten away by earlier failures *and* by
censoring. Censoring costs you nothing in bias (if non-informative) but
plenty in **variance** — you see that trade every time a survival curve's
band flares out at the tail.
"""
    )

T = np.array([4, 12, 7, 5, 11, 12])
E = np.array([1, 0, 1, 0, 1, 0])
names = ["M1", "M2", "M3", "M4", "M5", "M6"]

rows = []
S_hat = 1.0
km_t, km_s = [0.0], [1.0]
for ti in np.unique(T[E == 1]):
    n_i = int((T >= ti).sum())                  # still at risk
    d_i = int(((T == ti) & (E == 1)).sum())     # failed at ti
    S_hat *= (1 - d_i / n_i)
    rows.append({"failure time tᵢ": int(ti), "at risk nᵢ": n_i,
                 "failed dᵢ": d_i,
                 "survival fraction (1 − dᵢ/nᵢ)": f"{1 - d_i / n_i:.3f}",
                 "Ŝ(tᵢ) (running product)": f"{S_hat:.3f}"})
    km_t += [ti, ti]
    km_s += [km_s[-1], S_hat]
km_t.append(12)
km_s.append(km_s[-1])
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

kmf = KaplanMeierFitter().fit(T, E)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (nm, ti, ei) in enumerate(zip(names, T, E)):
    yy = 5 - i
    axes[0].plot([0, ti], [yy, yy], color="#546e7a", linewidth=2)
    axes[0].scatter([ti], [yy], marker="X" if ei else ">", s=110,
                    color="#c62828" if ei else "#2e7d32", zorder=5)
    axes[0].text(ti + 0.25, yy, "failed" if ei else "censored", fontsize=8,
                 va="center", color="#c62828" if ei else "#2e7d32")
    axes[0].text(-0.5, yy, nm, fontsize=9, ha="right", va="center")
axes[0].set_xlim(-1.6, 15)
axes[0].set_ylim(-0.6, 5.6)
axes[0].set_xlabel("months")
axes[0].set_yticks([])
axes[0].set_title("who failed, who was censored", fontsize=10)
axes[0].grid(alpha=0.2, axis="x")

axes[1].step(km_t, km_s, where="post", color="#1565c0", linewidth=2.4,
             label="our hand-computed KM")
kmf.plot_survival_function(ax=axes[1], color="#e65100", linestyle="--",
                           linewidth=1.6, ci_show=False,
                           label="lifelines KaplanMeierFitter")
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("months")
axes[1].set_ylabel("Ŝ(t)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)
axes[1].set_title("the Kaplan–Meier curve (a step function:\n"
                  "it only drops when someone actually fails)", fontsize=9)
st.pyplot(fig)
plt.close(fig)
st.success(
    "Our by-hand product and `lifelines` land on the same curve. Notice the "
    "curve **does not drop** at months 5 or 12 (the censoring times) — "
    "censoring never causes a drop, it just shrinks the at-risk set for "
    "later drops."
)

st.header("4 · The C-index: the metric this whole section is judged on")
st.markdown(
    r"""
Finally, how do we *score* a survival model? Not by accuracy or MSE — by the
**concordance index (C-index)**, and you already understand it, because on
the Model Evaluation page you met its ancestor: **AUC**.

$$C = P\big(\hat{r}_i > \hat{r}_j \;\big|\; T_i < T_j \big)$$

In words: **take a random pair of machines where we know which failed first;
what's the chance the model gave the earlier-failing one the higher risk
score?** It's a *ranking* metric — it never asks the model to predict a time,
only an ordering. C = 0.5 is a coin flip; C = 1.0 is a perfect ranking.

The censoring-aware twist: a pair is only **usable** (*comparable*) if we can
actually tell who failed first. If machine A was censored at month 3 and B
failed at month 8, we know B outlived A's *observation*, but not A's true
lifetime — so that pair is discarded. Concordance is computed over comparable
pairs only.

Every number in the DeepSurv paper — and every number in the rest of this
section — is a C-index. Here it is computed from scratch, and checked against
`lifelines`:
"""
)

with st.expander("🎓 Deeper statistics — the C-index is a U-statistic "
                 "(and cousin of Kendall's τ)"):
    st.markdown(
        r"""
Strip the censoring away for a second. "Probability a random pair is ranked
correctly" is exactly the estimand of the **Mann–Whitney U statistic**, and
the C-index is its estimator: an average of a 0/1 *kernel* over all pairs —
a **U-statistic**. That buys you real theory for free: U-statistics are
unbiased for their estimand and asymptotically normal, which is why papers
can put standard errors on C-indexes.

Three consequences worth carrying around:

1. **AUC is the special case** where "time" is binary (event/no event) — the
   same statistic, so your intuition for AUC (0.5 = coin flip, threshold-free,
   only ranks matter) transfers wholesale.
2. It's a linear transform of **Kendall's τ** between predicted risk and
   failure time ($C \approx (\tau + 1)/2$) — the C-index is rank correlation
   wearing a survival costume.
3. **Censoring makes the pair set data-dependent**: Harrell's C averages over
   *comparable* pairs only, and which pairs are comparable depends on the
   censoring distribution. So two studies of the same model with different
   censoring patterns get different C-indexes — a subtle non-comparability
   that the IPCW-weighted variants (Uno's C) exist to fix. The DeepHit page
   on evaluation returns to this with the time-dependent $C^{td}$.
"""
    )
show_example(
    '''import numpy as np
from lifelines.utils import concordance_index

# 6 machines: observed time, event (1=failed, 0=censored), model's risk score
T    = np.array([4, 12,  7,  5, 11, 12])
E    = np.array([1,  0,  1,  0,  1,  0])
risk = np.array([2.1, -0.5, 1.4, 0.3, 0.9, -1.2])   # higher = riskier

comparable, concordant = 0, 0
for i in range(len(T)):
    for j in range(len(T)):
        if i == j:
            continue
        # a pair is usable only if i FAILED and did so before j's last-seen time
        if E[i] == 1 and T[i] < T[j]:
            comparable += 1
            if risk[i] > risk[j]:      # did the model rank the earlier failure riskier?
                concordant += 1
            elif risk[i] == risk[j]:
                concordant += 0.5      # ties count half

print("comparable pairs:", comparable, " concordant:", concordant)
print("my C-index:      ", round(concordant / comparable, 4))
print("lifelines:       ", round(concordance_index(T, -risk, E), 4))''',
    """
- `risk` — the model's output. **Higher = expected to fail sooner.** This sign convention trips everyone up once.
- The double loop enumerates ordered pairs. The condition `E[i] == 1 and T[i] < T[j]` is the comparability rule: machine *i* must have genuinely **failed** (not been censored), and it must have done so before *j* was last seen. Only then do we know the true ordering.
- `risk[i] > risk[j]` — the model gets the pair right if it scored the earlier-failing machine as riskier. Ties earn half a point, by convention.
- `concordance_index(T, -risk, E)` — lifelines expects a *predicted survival time*-like quantity (higher = lives longer), so we pass `-risk`. Matching our by-hand number confirms both the maths and the sign convention. **Remember the minus sign** — you'll need it for every model in this section.
""",
)

guided_sandbox(
    key="d1",
    steps="""
1. **Step 1** — compute the Kaplan–Meier estimate by hand for the machines in
   the editor: loop over the sorted unique **failure** times, count `n_i`
   (`(T >= ti).sum()`) and `d_i` (`((T == ti) & (E == 1)).sum()`), and build
   the running product. Print a row per failure time.
2. **Step 2** — check it against `KaplanMeierFitter().fit(T, E)`
   (`kmf.survival_function_`). Do your numbers match?
3. **Step 3** — what fraction of these machines are censored? Print it, then
   print the **naive** mean lifetime you'd get by wrongly treating every T as
   a true failure time, and compare it to the mean of the *uncensored* ones
   only. Neither is right — convince yourself why.
4. **Step 4 (stretch)** — write `c_index(T, E, risk)` from scratch with the
   comparable-pairs rule, and check it against
   `concordance_index(T, -risk, E)`. Then try `risk = -risk` and watch the
   C-index flip to `1 - C`. That's the sign convention biting.
""",
    setup_code='''import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

# 12 machines on the test rig
T    = np.array([4, 12, 7, 5, 11, 12, 2, 9, 12, 6, 3, 12])
E    = np.array([1,  0, 1, 0,  1,  0, 1, 1,  0, 1, 1,  0])
risk = np.array([2.1, -0.5, 1.4, 0.3, 0.9, -1.2,
                 2.8, 0.6, -0.9, 1.1, 2.4, -0.7])
print("n =", len(T), " failures =", E.sum(), " censored =", (E == 0).sum())

# Step 1: Kaplan-Meier by hand


# Step 2: compare with lifelines KaplanMeierFitter


# Step 3: censoring fraction; naive mean vs uncensored-only mean


# Step 4 (stretch): c_index(T, E, risk) from scratch, checked vs lifelines
''',
    solution_code='''import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

T    = np.array([4, 12, 7, 5, 11, 12, 2, 9, 12, 6, 3, 12])
E    = np.array([1,  0, 1, 0,  1,  0, 1, 1,  0, 1, 1,  0])
risk = np.array([2.1, -0.5, 1.4, 0.3, 0.9, -1.2,
                 2.8, 0.6, -0.9, 1.1, 2.4, -0.7])

S = 1.0
print("t_i  n_i  d_i   S(t_i)")
for ti in np.unique(T[E == 1]):
    n_i = int((T >= ti).sum())
    d_i = int(((T == ti) & (E == 1)).sum())
    S *= (1 - d_i / n_i)
    print(f"{ti:3d} {n_i:4d} {d_i:4d}   {S:.4f}")

kmf = KaplanMeierFitter().fit(T, E)
print("\\nlifelines:")
print(kmf.survival_function_.round(4).to_string())

print(f"\\ncensored: {(E == 0).mean():.0%}")
print(f"naive mean of all T:        {T.mean():.2f} months (WRONG: counts "
      "censored times as failures -> too SHORT)")
print(f"mean of uncensored only:    {T[E == 1].mean():.2f} months (WRONG: "
      "throws away the longest-lasting machines -> also too short)")

def c_index(T, E, risk):
    comparable = concordant = 0.0
    for i in range(len(T)):
        for j in range(len(T)):
            if i != j and E[i] == 1 and T[i] < T[j]:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5
    return concordant / comparable

print(f"\\nmy C-index:        {c_index(T, E, risk):.4f}")
print(f"lifelines C-index: {concordance_index(T, -risk, E):.4f}")
print(f"with the sign flipped: {c_index(T, E, -risk):.4f} (= 1 - C)")''',
)
