import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter
from matplotlib.patches import Rectangle

from utils.sandbox import guided_sandbox, show_example

st.title("📐 The Cox Proportional Hazards Model")
st.markdown(
    r"""
Now we model. We want the hazard of a *particular* machine to depend on its
covariates $x$ (sensor readings, age, load). Cox's 1972 proposal — one of the
most-cited papers in the history of statistics — is:

$$\boxed{\;h(t \mid x) = \underbrace{h_0(t)}_{\text{baseline hazard}} \cdot \underbrace{\exp(\beta^\top x)}_{\text{risk score}}\;}$$

Read the structure carefully, because **DeepSurv changes exactly one thing in
this equation and nothing else**:

- $h_0(t)$ — the **baseline hazard**: how risk evolves over time for a
  "reference" machine ($x = 0$). It's an arbitrary, unspecified function of
  time. **Cox never estimates it**, which is the model's genius.
- $\exp(\beta^\top x)$ — a **risk score** depending only on the covariates,
  never on time. It scales the whole baseline curve up or down.

The name comes from the immediate consequence. Take two machines and divide:

$$\frac{h(t \mid x_1)}{h(t \mid x_2)} = \frac{h_0(t)\exp(\beta^\top x_1)}{h_0(t)\exp(\beta^\top x_2)} = \exp\!\big(\beta^\top (x_1 - x_2)\big)$$

The baseline **cancels**. The hazard ratio between any two machines is a
constant — *it does not depend on $t$*. Machine A is 2.3× riskier than
machine B at month 1, at month 50, forever. That's the **proportional
hazards assumption**, and Section 7 exists because it is often false.
"""
)

st.header("1 · The partial likelihood — and the risk-set walk")
st.markdown(
    r"""
Here's the problem Cox had to solve. The full likelihood needs $h_0(t)$,
which we refuse to specify. His escape was the **partial likelihood**: at each
moment a failure occurs, ask a question whose answer *doesn't need* the
baseline.

> "Machine $i$ failed at time $T_i$. Given that **exactly one** of the
> machines still at risk failed at this instant, what's the probability it
> was $i$ specifically?"

Because every machine at risk shares the same $h_0(T_i)$, it cancels top and
bottom, leaving:

$$L_i(\beta) = \frac{\exp(\beta^\top x_i)} {\sum_{j \in \mathcal{R}(T_i)} \exp(\beta^\top x_j)}$$

where $\mathcal{R}(T_i) = \{j : T_j \ge T_i\}$ is the **risk set** — every
machine still being watched at that moment. Multiply over all observed
failures:

$$L(\beta) = \prod_{i:\,E_i = 1} L_i(\beta), \qquad \ell(\beta) = \sum_{i:\,E_i=1}\Big[\beta^\top x_i - \log \!\!\sum_{j \in \mathcal{R}(T_i)}\!\! e^{\beta^\top x_j}\Big]$$

**As a statistician, notice what this really is:** each $L_i$ is exactly a
*softmax* — the probability of picking machine $i$ out of the risk set, with
$\beta^\top x$ as the logits. Cox's partial likelihood is a chain of
conditional multinomial choices. (You met softmax on the activations page.
This is not a coincidence, and it's why the whole thing ports so cleanly to a
neural network.)

**Walk through it below.** Each step is one observed failure; the risk set is
computed live from the data, and so is that failure's contribution to the
log-partial-likelihood.
"""
)

rng = np.random.default_rng(5)
n = 9
load = rng.uniform(1, 9, n).round(1)         # a single covariate: machine load
beta_true = 0.35
times = rng.exponential(scale=np.exp(-beta_true * (load - 5))) * 6
times = np.round(times + 1, 1)
event = (rng.uniform(0, 1, n) > 0.3).astype(int)
labels = [f"M{i + 1}" for i in range(n)]
df = pd.DataFrame({"machine": labels, "load": load, "T": times, "E": event})

order = np.argsort(times)
failure_order = [i for i in order if event[i] == 1]

beta = st.slider("β — the coefficient on 'load' (try different values)",
                 -0.6, 0.9, 0.35, 0.05)
step = st.slider("walk through the observed failures", 1, len(failure_order),
                 1)

fig, ax = plt.subplots(figsize=(10, 4.4))
current_i = failure_order[step - 1]
t_i = times[current_i]
risk_set = [j for j in range(n) if times[j] >= t_i]

for rank, j in enumerate(order):
    yy = n - 1 - rank
    in_risk = j in risk_set
    is_current = j == current_i
    ax.plot([0, times[j]], [yy, yy],
            color="#2e7d32" if is_current else ("#546e7a" if in_risk
                                                else "#cfd8dc"),
            linewidth=3 if is_current else 2)
    ax.scatter([times[j]], [yy], marker="X" if event[j] else ">", s=130,
               color="#c62828" if event[j] else "#2e7d32",
               alpha=1.0 if in_risk else 0.3, zorder=5)
    ax.text(-0.3, yy, f"{labels[j]} (load {load[j]:.1f})", fontsize=8,
            ha="right", va="center",
            color="#263238" if in_risk else "#b0bec5",
            fontweight="bold" if is_current else "normal")
    if in_risk:
        ax.text(times[j] + 0.25, yy,
                f"e^(β·x)={np.exp(beta * load[j]):.2f}", fontsize=7.5,
                va="center", color="#37474f")
ax.axvline(t_i, color="#c62828", linestyle="--", linewidth=2)
ax.add_patch(Rectangle((t_i, -0.6), max(times) - t_i + 3, n + 0.2,
                       facecolor="#fff9c4", alpha=0.35, zorder=0))
ax.text(t_i, n - 0.3, f"  failure at T={t_i}", fontsize=9, color="#c62828")
ax.text(t_i + 0.2, -0.45, "the RISK SET: everyone still being watched here",
        fontsize=8, color="#f57f17")
ax.set_xlim(-4.2, max(times) + 3.5)
ax.set_ylim(-0.8, n + 0.3)
ax.set_yticks([])
ax.set_xlabel("months")
ax.set_title(f"failure {step} of {len(failure_order)}: "
             f"{labels[current_i]} fails — who else was still at risk?",
             fontsize=11)
st.pyplot(fig)
plt.close(fig)

num = np.exp(beta * load[current_i])
den = sum(np.exp(beta * load[j]) for j in risk_set)
contrib = np.log(num) - np.log(den)
running = 0.0
for k in range(step):
    ii = failure_order[k]
    rs = [j for j in range(n) if times[j] >= times[ii]]
    running += (beta * load[ii]
                - np.log(sum(np.exp(beta * load[j]) for j in rs)))

c1, c2, c3 = st.columns(3)
c1.metric("risk set size |R(Tᵢ)|", len(risk_set))
c2.metric("this failure's Lᵢ", f"{num / den:.3f}",
          help="the softmax probability that THIS machine was the one to fail")
c3.metric("running log partial likelihood", f"{running:.3f}")
st.markdown(
    f"""
**{labels[current_i]}** failed at T={t_i}. The partial likelihood asks: *of
the {len(risk_set)} machines still at risk, what was the chance it would be
this one?*

$$L_i = \\frac{{e^{{\\beta \\cdot {load[current_i]:.1f}}}}} {{\\sum_{{j \\in R}} e^{{\\beta \\cdot x_j}}}} = \\frac{{{num:.3f}}}{{{den:.3f}}} = {num / den:.3f}$$

Three things to *do* with the sliders, because they build the intuition the
rest of this section rests on:

- **Walk the failures.** The risk set shrinks monotonically — machines
  leave by failing or by being censored, and never come back. Censored
  machines (green arrows) sit in risk sets right up to the moment they
  leave: **that is how they contribute to the estimate despite never
  failing.**
- **Drag β up.** High-load machines get bigger $e^{{\\beta x}}$. If the
  high-load machines really are the ones failing early, the numerator grows
  faster than the denominator and the log-likelihood rises. Maximising this
  quantity over β *is* fitting the model.
- **Note the last failure.** When the risk set has only one member left,
  $L_i = 1$ and it contributes exactly 0 — no information. The early
  failures, with big risk sets, carry the most.
"""
)

st.header("2 · Fitting it: maximum likelihood, exactly as you were taught")
st.markdown(
    r"""
There's no closed form for $\hat\beta$, so we maximise $\ell(\beta)$
numerically — which for you means: **this is just MLE, and everything you
know about MLE still applies.** Standard errors come from the inverse
observed information (the Hessian of $-\ell$), Wald tests and confidence
intervals follow as usual, likelihood-ratio tests compare nested models, and
**AIC** = $2k - 2\ell(\hat\beta)$ is available for model comparison (that's
the `AIC_` attribute on a fitted lifelines model — the bridge promised back
on the Model Evaluation page).

Below: our own gradient ascent on the partial likelihood, checked against
lifelines' fitted β.
"""
)
show_example(
    '''import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

rng = np.random.default_rng(5)
n = 200
load = rng.uniform(1, 9, n)
true_beta = 0.35
T = rng.exponential(scale=np.exp(-true_beta * (load - 5))) * 6 + 1
E = (rng.uniform(0, 1, n) > 0.3).astype(int)

def log_partial_likelihood(beta):
    ll = 0.0
    for i in np.where(E == 1)[0]:
        at_risk = T >= T[i]                       # the risk set
        ll += beta * load[i] - np.log(np.sum(np.exp(beta * load[at_risk])))
    return ll

# maximise by simple gradient ascent (nudge-test the gradient, as always)
beta = 0.0
for step in range(300):
    g = (log_partial_likelihood(beta + 1e-5)
         - log_partial_likelihood(beta - 1e-5)) / 2e-5
    beta += 0.0002 * g
print(f"my gradient ascent:  beta = {beta:.4f}")

df = pd.DataFrame({"load": load, "T": T, "E": E})
cph = CoxPHFitter().fit(df, "T", "E")
print(f"lifelines:           beta = {cph.params_['load']:.4f}")
print(f"the TRUE beta was:          {true_beta}")
print(f"\\nhazard ratio per unit of load: {np.exp(cph.params_['load']):.3f}")
print(f"log partial likelihood at the fit: {cph.log_likelihood_:.3f}")
print(f"AIC: {cph.AIC_partial_:.2f}")''',
    """
- `at_risk = T >= T[i]` — the risk set as a boolean mask (NumPy page!). Everything with an observed or censored time at least as large as `T[i]` is still being watched.
- `ll += beta * load[i] - np.log(...)` — the log-partial-likelihood formula, transcribed directly. The sum runs **only over failures** (`np.where(E == 1)`), but the risk set inside includes **censored machines too**.
- The gradient-ascent loop is Section 2's algorithm with the sign flipped (we're *maximising* a likelihood, so we step *up*). We use the finite-difference gradient — the same nudge test you used to verify backprop.
- Our β and lifelines' β agree, and both land near the true 0.35 that generated the data. Two independent routes, one answer.
- `np.exp(beta)` — the **hazard ratio**: each extra unit of load multiplies the failure hazard by this factor. This exponentiated-coefficient interpretation is why Cox models are so beloved in applied work — and it's exactly what you lose when you go deep.
- `cph.AIC_partial_` — AIC on the partial likelihood, for comparing Cox models. (lifelines names it `AIC_partial_` for the Cox model specifically.)
""",
)

with st.expander("🎓 Deeper statistics — why a *partial* likelihood earns "
                 "full-likelihood treatment, and how you get $S(t\\mid x)$ "
                 "back at the end"):
    st.markdown(
        r"""
**The inference question.** We just used Wald tests, information-based
standard errors and AIC on something that is *not* the likelihood of the
data — pieces of it were thrown away (the actual failure *times*, the gaps
between them, everything about $h_0$). Why is that legitimate?

Cox's 1975 argument, in modern language: the log partial likelihood's
derivative behaves exactly like a genuine **score function** — it has mean
zero at the true $\beta$, and its variance equals the expected negative
Hessian (the **information identity** you proved for ordinary MLE). Those
two properties are what drive the standard MLE asymptotics, so the usual
conclusions transfer:
$\hat\beta \xrightarrow{d} N\big(\beta,\, \mathcal{I}(\hat\beta)^{-1}\big)$,
Wald/score/LR tests all valid. Later work sharpened this: the partial
likelihood estimator is **semiparametrically efficient** — no estimator that
also refuses to model $h_0$ can beat its asymptotic variance. Discarding
the timing information costs remarkably little, because the *ordering* of
failures carries almost all the information about $\beta$.

Two ways to see why it deserves the name "likelihood":

- **Profile view:** write the full censored-data likelihood from the last
  page, maximise over the infinite-dimensional nuisance $h_0(\cdot)$ for
  fixed $\beta$, and plug the maximiser back in. What remains is (up to a
  constant) the partial likelihood. So $\ell(\beta)$ *is* a profile
  likelihood with the baseline profiled out.
- **Rank view:** the partial likelihood is the probability of the observed
  *order* of failures. $\beta$ only ever enters through orderings — which
  is also exactly why the natural performance metric (C-index) is a rank
  statistic.

**Recovering absolute risk (Breslow's estimator).** The model happily ranks
machines without $h_0$, but the moment you want an actual survival curve
$S(t \mid x)$ you must estimate the baseline after the fact:

$$\hat H_0(t) = \sum_{i:\,T_i \le t,\,E_i=1} \frac{1}{\sum_{j \in \mathcal{R}(T_i)} e^{\hat\beta^\top x_j}}, \qquad \hat S(t \mid x) = \exp\big(-\hat H_0(t)\, e^{\hat\beta^\top x}\big)$$

Look at the structure: it's a Nelson–Aalen-style cumulative hazard where
each failure counts not as $1/n_i$ but as $1$ over the risk set's *total
risk score* — KM logic, reweighted by the fitted model. **File this away:**
DeepSurv inherits the same two-stage recipe (rank with the network, then
Breslow for the baseline), and `pycox` calls it `compute_baseline_hazards()`.
DeepHit's whole pitch on the next section is to skip this second stage and
predict the curve directly.
"""
    )

st.header("3 · The limitation that created DeepSurv")
st.markdown(
    r"""
Look hard at the risk score: $\exp(\beta^\top x)$. The log-risk
$\beta^\top x$ is **linear** in the covariates. That builds in three
assumptions, and real machinery (and real patients) violate all three:

1. **Each covariate acts monotonically and log-linearly.** Doubling a
   sensor reading doubles its contribution to log-risk. But suppose a machine
   is safest at a *moderate* operating temperature and fails at both extremes
   — a U-shaped risk. A linear term cannot represent that at all; the fitted
   β will come out near zero, and the model will confidently report that
   temperature "doesn't matter". **This is precisely the simulation the
   DeepSurv paper runs, and precisely why Cox scores C ≈ 0.49 (a coin flip)
   on it** — you'll reproduce that yourself two pages from now.
2. **No interactions unless you write them yourself.** If high load is only
   dangerous *when* the coolant is weak, you must hand-specify
   `load × coolant` as a term. With 40 covariates there are 780 possible
   pairwise interactions and you must guess which matter.
3. **Proportional hazards.** The hazard ratio between two machines is
   constant for all time — an assumption you can (and should) test, e.g. with
   Schoenfeld residuals.

The classical fixes are the ones you'd reach for in a stats course: add
polynomial terms, splines, or hand-picked interactions. They work — if you
know what to add. **DeepSurv's proposal is to stop guessing:** replace
$\beta^\top x$ with a neural network $\hat h_\theta(x)$ and let it learn the
shape from the data. That is the *entire* idea, and the next three pages build
it.

Note carefully what DeepSurv does **not** change: the model is *still*
$h(t|x) = h_0(t)\exp(\hat h_\theta(x))$ — still proportional hazards, still no
baseline estimated, still fitted by maximising the same partial likelihood.
Assumption 3 survives untouched. **That's what DeepHit will attack in
Section 7.**
"""
)

guided_sandbox(
    key="d2",
    steps="""
1. **Step 1** — write `log_partial_likelihood(beta)`: loop over the indices
   where `E == 1`, build the risk-set mask `T >= T[i]`, and accumulate
   `beta * load[i] - np.log(np.sum(np.exp(beta * load[at_risk])))`.
2. **Step 2** — evaluate it for `beta` in `np.linspace(-0.5, 1.0, 16)`, print
   each, and find the maximising β with `np.argmax`. You have just done MLE
   by brute force.
3. **Step 3** — plot is optional; instead compare your best β with
   `CoxPHFitter().fit(df, "T", "E").params_["load"]` and with the true β
   (0.35) that generated the data. Also print the fitted **hazard ratio**
   `np.exp(beta)` and say in words what it means.
4. **Step 4 (stretch)** — the U-shape trap. Regenerate the data so that risk
   depends on `(load - 5) ** 2` instead of `load` (code hint in the editor),
   refit a Cox model on the **raw `load`** column, and look at the p-value.
   Cox will tell you load doesn't matter. Then add a `load_sq` column and
   refit — suddenly it's highly significant. That gap is DeepSurv's entire
   reason for existing.
""",
    setup_code='''import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

rng = np.random.default_rng(5)
n = 300
load = rng.uniform(1, 9, n)
true_beta = 0.35
T = rng.exponential(scale=np.exp(-true_beta * (load - 5))) * 6 + 1
E = (rng.uniform(0, 1, n) > 0.3).astype(int)
df = pd.DataFrame({"load": load, "T": T, "E": E})
print(f"{n} machines, {E.sum()} observed failures, {(E==0).sum()} censored")

# Step 1: define log_partial_likelihood(beta)


# Step 2: brute-force MLE over a grid of beta values


# Step 3: compare with lifelines; print the hazard ratio


# Step 4 (stretch): the U-shaped risk trap
#   T2 = rng.exponential(scale=np.exp(-0.4 * (load - 5) ** 2)) * 6 + 1
#   fit Cox on raw `load` -> insignificant. Add load_sq -> significant.
''',
    solution_code='''import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

rng = np.random.default_rng(5)
n = 300
load = rng.uniform(1, 9, n)
true_beta = 0.35
T = rng.exponential(scale=np.exp(-true_beta * (load - 5))) * 6 + 1
E = (rng.uniform(0, 1, n) > 0.3).astype(int)
df = pd.DataFrame({"load": load, "T": T, "E": E})

def log_partial_likelihood(beta):
    ll = 0.0
    for i in np.where(E == 1)[0]:
        at_risk = T >= T[i]
        ll += beta * load[i] - np.log(np.sum(np.exp(beta * load[at_risk])))
    return ll

grid = np.linspace(-0.5, 1.0, 16)
lls = [log_partial_likelihood(b) for b in grid]
best = grid[int(np.argmax(lls))]
for b, ll in list(zip(grid, lls))[::3]:
    print(f"beta={b:+.2f}: log partial likelihood {ll:.2f}")
print(f"\\nbrute-force MLE:  beta = {best:.3f}")

cph = CoxPHFitter().fit(df, "T", "E")
print(f"lifelines:        beta = {cph.params_['load']:.3f}")
print(f"true beta:               {true_beta}")
print(f"hazard ratio: {np.exp(cph.params_['load']):.3f} -> each extra unit "
      "of load multiplies the failure hazard by this")

print("\\n--- the U-shape trap ---")
T2 = rng.exponential(scale=np.exp(-0.4 * (load - 5) ** 2)) * 6 + 1
E2 = (rng.uniform(0, 1, n) > 0.3).astype(int)
df2 = pd.DataFrame({"load": load, "T": T2, "E": E2})
c2 = CoxPHFitter().fit(df2, "T", "E")
print(f"Cox on raw load:   beta={c2.params_['load']:+.3f}, "
      f"p={c2.summary.loc['load', 'p']:.3f}  <- 'load does not matter'")

df2["load_sq"] = (load - 5) ** 2
c3 = CoxPHFitter().fit(df2, "T", "E")
print(f"Cox with load_sq:  beta={c3.params_['load_sq']:+.3f}, "
      f"p={c3.summary.loc['load_sq', 'p']:.2e}  <- it mattered enormously")
print("Cox needed a human to guess the right transformation. "
      "DeepSurv learns it.")''',
)
