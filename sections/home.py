import streamlit as st

st.title("🎓 Python → Machine Learning → Survival Analysis")
st.markdown(
    """
Welcome! This app is a complete, self-contained course that takes you from
basic Python all the way to a deep understanding of the two research papers
behind your dissertation:

- **Katzman et al. 2018 — DeepSurv** (*BMC Medical Research Methodology*)
- **Lee, Zame, Yoon & van der Schaar 2018 — DeepHit** (*AAAI*)

### How every page works

Each topic page gives you the same three things:

1. **A plain-English explanation** — written for someone who knows variables,
   loops, if/else and functions, and nothing else. Later pages build only on
   earlier pages.
2. **A visualization of the mechanism** — not a chart of results, but a
   picture of the concept *happening*: centroids moving, a cursor walking
   through a file, numbers flowing through a network.
3. **🧪 A "try it yourself" code sandbox** — a real editor pre-filled with
   working code. Edit it, press **Ctrl+Enter** (or the **APPLY** button in
   the editor's corner), and the output below updates instantly. You never
   need to leave the app or open a terminal.

### A promise about honesty

Every output box in this course is **computed live on your machine** when the
page loads — nothing is pasted in from memory. If something on a page could
not be verified by actually running it, the page says so in a visible note.

### The curriculum

| # | Section | Status |
|---|---------|--------|
| 1 | Python Fundamentals | ✅ Ready |
| 2 | ML Foundations | ✅ Ready |
| 3 | Classical ML Algorithms | ✅ Ready |
| 4 | PySpark & Big Data | ✅ Ready |
| 5 | Neural Networks | ✅ Ready |
| 6 | DeepSurv, in full | ✅ Ready |
| 7 | DeepHit, in full | ✅ Ready |

Use the sidebar on the left to navigate. Start with
**1 · Python Fundamentals → Data Structures & Comprehensions** and work
downward in order — each page assumes only what came before it.
"""
)
