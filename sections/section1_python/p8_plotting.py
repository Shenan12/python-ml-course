import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.sandbox import sandbox, show_example

st.title("📊 Plotting Fundamentals")
st.markdown(
    """
A statistician who can't plot is flying blind — and in ML you'll plot *loss
curves*, *decision boundaries* and *survival curves* constantly from Section 2
onward. We use **matplotlib**, the standard Python plotting library, plus a
taste of **plotly** for interactive charts.

The one idea that unlocks matplotlib: you build a **Figure** (the canvas)
containing one or more **Axes** (an actual plot with an x-axis and y-axis),
and then call methods *on the axes* to draw. `fig, ax = plt.subplots()` hands
you both.
"""
)

st.header("1 · The anatomy of a plot — toggle each part on and off")
st.markdown(
    "Every part of a chart has a name and one line of code that creates it. "
    "Untick a box and the corresponding line is genuinely skipped when the "
    "figure below is rebuilt:"
)

c = st.columns(5)
want_title = c[0].checkbox("ax.set_title(...)", True)
want_labels = c[1].checkbox("ax.set_x/ylabel(...)", True)
want_legend = c[2].checkbox("ax.legend()", True)
want_grid = c[3].checkbox("ax.grid(...)", True)
want_markers = c[4].checkbox("marker='o'", True)

weeks = np.arange(1, 11)
newton = np.array([62, 64, 63, 67, 70, 69, 72, 74, 73, 76])
curie = np.array([70, 69, 72, 71, 74, 76, 75, 78, 80, 79])

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(weeks, newton, color="#1565c0", label="Newton",
        marker="o" if want_markers else None)
ax.plot(weeks, curie, color="#e65100", label="Curie",
        marker="s" if want_markers else None)
if want_title:
    ax.set_title("Average mark per week, by house")
    ax.annotate("the title — set_title()", xy=(0.5, 1.01),
                xycoords="axes fraction", xytext=(0.14, 1.06),
                fontsize=8, color="#8e24aa",
                arrowprops=dict(arrowstyle="->", color="#8e24aa"))
if want_labels:
    ax.set_xlabel("week of term")
    ax.set_ylabel("average mark")
    ax.annotate("axis label — set_xlabel()", xy=(0.55, -0.09),
                xycoords="axes fraction", xytext=(0.72, -0.16),
                fontsize=8, color="#8e24aa",
                arrowprops=dict(arrowstyle="->", color="#8e24aa"))
if want_legend:
    ax.legend(loc="upper left")
    ax.annotate("the legend — legend()", xy=(0.13, 0.93),
                xycoords="axes fraction", xytext=(0.3, 0.97),
                fontsize=8, color="#8e24aa",
                arrowprops=dict(arrowstyle="->", color="#8e24aa"))
if want_grid:
    ax.grid(alpha=0.3)
if want_markers:
    ax.annotate("a marker — marker='o'", xy=(4, newton[3]),
                xytext=(4.6, 63.5), fontsize=8, color="#8e24aa",
                arrowprops=dict(arrowstyle="->", color="#8e24aa"))
st.pyplot(fig)
plt.close(fig)

st.header("2 · The code behind it, line by line")
show_example(
    '''import matplotlib.pyplot as plt
import numpy as np

weeks  = np.arange(1, 11)
newton = np.array([62, 64, 63, 67, 70, 69, 72, 74, 73, 76])
curie  = np.array([70, 69, 72, 71, 74, 76, 75, 78, 80, 79])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(weeks, newton, marker="o", label="Newton")
ax.plot(weeks, curie,  marker="s", label="Curie")
ax.set_title("Average mark per week, by house")
ax.set_xlabel("week of term")
ax.set_ylabel("average mark")
ax.legend()
ax.grid(alpha=0.3)
plt.show()''',
    """
- `np.arange(1, 11)` — the numbers 1…10 (stop excluded, as ever): our x-axis.
- `fig, ax = plt.subplots(figsize=(8, 4))` — create one Figure containing one Axes; `figsize` is (width, height) in inches. The `fig, ax = ...` is tuple unpacking from the Functions page.
- `ax.plot(x, y, marker="o", label="Newton")` — draw a line through the (x, y) points. `marker="o"` puts a dot at each data point; `label=` is the name the legend will show. Note `marker` and `label` are **keyword arguments** being scooped up exactly like the `**kwargs` you met earlier.
- Calling `ax.plot` twice — draws a second line *on the same axes*; matplotlib picks a new colour automatically.
- `ax.set_title / set_xlabel / set_ylabel` — a chart without labelled axes is unreadable; in your dissertation these are non-negotiable.
- `ax.legend()` — builds the key from the `label=`s you provided.
- `ax.grid(alpha=0.3)` — faint gridlines; `alpha` is transparency (0 invisible → 1 solid).
- `plt.show()` — displays the figure in a normal script. (Inside this app the sandbox captures figures automatically, so it's optional here.)
""",
)

st.header("3 · Which chart answers which question?")
chart = st.radio(
    "Pick the *question* you're asking of the data",
    ["Trend over time → line", "Relationship between two variables → scatter",
     "Comparison across categories → bar",
     "Shape of a distribution → histogram"],
    horizontal=False,
)

rng = np.random.default_rng(42)
study_hours = rng.uniform(0, 12, 60)
marks = np.clip(40 + 4 * study_hours + rng.normal(0, 8, 60), 0, 100)
all_marks = np.clip(rng.normal(66, 12, 200), 0, 100)
# The same 9 students as the Pandas page; the means are computed live here:
students = pd.DataFrame({
    "house": ["Newton", "Curie", "Newton", "Turing", "Curie", "Turing",
              "Newton", "Curie", "Turing"],
    "mark": [68, 91, 75, 55, 82, 61, 88, 73, 70],
})
house_means = students.groupby("house")["mark"].mean()[
    ["Newton", "Curie", "Turing"]]

fig, ax = plt.subplots(figsize=(8, 3.8))
if chart.endswith("line"):
    ax.plot(weeks, newton, marker="o", color="#1565c0")
    ax.set_xlabel("week of term")
    ax.set_ylabel("average mark")
    ax.set_title("LINE: how does a value move as time passes?")
    verdict = ("A line implies the x-axis has an *order* (time, epochs, "
               "dosage). Your training-loss curves in Section 5 will be "
               "exactly this chart.")
elif chart.endswith("scatter"):
    ax.scatter(study_hours, marks, color="#00897b", alpha=0.75)
    ax.set_xlabel("hours studied per week")
    ax.set_ylabel("final mark")
    ax.set_title("SCATTER: do two variables move together?")
    verdict = ("Each dot is one student — 60 simulated students, generated "
               "with a real upward rule plus noise (seed 42). The upward "
               "drift you can see *is* the correlation.")
elif chart.endswith("bar"):
    ax.bar(house_means.index, house_means.values,
           color=["#1565c0", "#ad1457", "#2e7d32"])
    ax.set_ylabel("mean mark")
    ax.set_title("BAR: how do categories compare?")
    verdict = ("Categories have no order, so a line would be misleading — "
               "bars keep them honestly separate. (These are the real "
               "Newton/Curie/Turing means from the Pandas page's groupby.)")
else:
    ax.hist(all_marks, bins=20, color="#5e35b1", edgecolor="white")
    ax.set_xlabel("mark")
    ax.set_ylabel("number of students")
    ax.set_title("HISTOGRAM: what values are common vs rare?")
    verdict = ("200 simulated marks chopped into 20 bins; each bar counts "
               "how many landed in that bin. As a statistician you know "
               "this shape well — and you'll draw histograms of *survival "
               "times* in Section 6.")
ax.grid(alpha=0.25)
st.pyplot(fig)
plt.close(fig)
st.info(verdict)

st.header("4 · A taste of plotly: charts you can hover")
st.markdown(
    "Matplotlib makes *pictures*; **plotly** makes charts that respond to "
    "your mouse. Hover over the dots below — the tooltip shows each "
    "student's exact values. We'll use plotly later whenever hovering helps "
    "(e.g. inspecting individual patients on a survival curve)."
)
df_plotly = pd.DataFrame({"hours_studied": study_hours.round(1),
                          "mark": marks.round(0)})
import plotly.express as px

fig_px = px.scatter(df_plotly, x="hours_studied", y="mark",
                    title="Same scatter, but hoverable (plotly)")
st.plotly_chart(fig_px, width="stretch")
st.caption(
    "The code: `px.scatter(df, x=\"hours_studied\", y=\"mark\")` — plotly "
    "express takes a DataFrame and column *names*, another payoff from "
    "learning pandas."
)

sandbox(
    '''import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(7)
hours = rng.uniform(0, 12, 80)
marks = np.clip(40 + 4 * hours + rng.normal(0, 8, 80), 0, 100)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(hours, marks, alpha=0.7, color="teal", label="students")
ax.set_xlabel("hours studied")
ax.set_ylabel("mark")
ax.set_title("Study time vs mark (simulated)")
ax.legend()
ax.grid(alpha=0.3)

# Challenge 1: change the color, and try marker="x" in the scatter call.
# Challenge 2: add the "perfect study" line:  ax.plot(x, 40 + 4*x)
#              with x = np.linspace(0, 12, 50), labelled "true rule".
# Challenge 3: swap the scatter for  ax.hist(marks, bins=15)  and
#              relabel the axes so the chart still tells the truth.''',
    key="p8",
)
