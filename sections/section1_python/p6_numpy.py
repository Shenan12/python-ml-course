import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from utils.sandbox import sandbox, show_example

st.title("🔢 NumPy: Arrays, Broadcasting & Vectorization")
st.markdown(
    """
**NumPy** is the foundation stone of every ML library you will ever touch —
pandas, scikit-learn and PyTorch are all built on it. It provides one thing:
the **array**, a grid of numbers that Python can do maths on *all at once*
instead of one element at a time. That "all at once" is called
**vectorization**, and it's not just tidier — it's dramatically faster.
"""
)

st.header("1 · The speed difference, measured on YOUR machine right now")


@st.cache_data(show_spinner="Timing the loop vs NumPy race…")
def race(n=300_000):
    values = list(range(n))
    t0 = time.perf_counter()
    squares_loop = []
    for v in values:                 # plain Python: one element at a time
        squares_loop.append(v * v)
    t_loop = time.perf_counter() - t0

    arr = np.arange(n)
    t0 = time.perf_counter()
    squares_np = arr * arr           # NumPy: the whole array in one go
    t_np = time.perf_counter() - t0
    assert squares_np[123] == squares_loop[123]  # same answer, honest race
    return t_loop, t_np


t_loop, t_np = race()
speedup = t_loop / t_np

c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(6, 2.6))
    bars = ax.barh(["Python for-loop", "NumPy  arr * arr"],
                   [t_loop * 1000, t_np * 1000],
                   color=["#ef9a9a", "#81c784"], edgecolor="#37474f")
    for bar, t in zip(bars, [t_loop, t_np]):
        ax.text(bar.get_width() * 1.02, bar.get_y() + 0.36,
                f"{t * 1000:.2f} ms", fontsize=10, va="center")
    ax.set_xlabel("time to square 300,000 numbers (milliseconds)")
    ax.set_xlim(0, t_loop * 1000 * 1.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with c2:
    st.metric("Measured speedup on this laptop",
              f"{speedup:.0f}× faster")
    st.caption(
        "These two timings were measured live by this page (and cached). "
        "Rerun with a cleared cache and the exact numbers will wobble — "
        "the *ratio's order of magnitude* is the stable, honest takeaway."
    )

st.markdown(
    """
Why so fast? A Python loop re-checks the type of every single element, every
time. NumPy stores all elements as the *same* type packed tightly in memory
and hands the whole operation to optimised, compiled machine code (with the
loop happening in C, not Python).
"""
)

st.header("2 · Arrays: creating them and doing maths on them")
show_example(
    '''import numpy as np

marks = np.array([68, 91, 75, 55, 82])
print("the array:", marks)
print("shape:", marks.shape, "| dtype:", marks.dtype)

# Maths applies to EVERY element at once - no loop written by you:
curved = marks + 5
print("everyone +5:", curved)

# Comparisons give an array of True/False - a "boolean mask":
passed = marks >= 70
print("mask:       ", passed)

# A mask used as an index KEEPS only the True positions:
print("marks >= 70:", marks[passed])

# 2-D arrays are matrices; you'll live in these:
grid = np.array([[1, 2, 3], [4, 5, 6]])
print("grid shape:", grid.shape)
print("column means:", grid.mean(axis=0))
print("row means:   ", grid.mean(axis=1))''',
    """
- `np.array([...])` — turns a list into an array. All elements get one shared type.
- `.shape` — the size along each dimension, as a tuple: `(5,)` means 1-D with 5 elements. `.dtype` is that shared element type (e.g. `int64`).
- `marks + 5` — **vectorized arithmetic**: NumPy adds 5 to every element. No `for` loop in sight.
- `marks >= 70` — vectorized comparison: an array of `True`/`False`, one per element, called a **boolean mask**.
- `marks[passed]` — indexing with a mask filters the array: only positions where the mask is `True` survive. This is *the* pattern for "select the rows where…" in all of data science.
- `grid.mean(axis=0)` — `axis=0` means "collapse the rows, give me one value per **column**"; `axis=1` collapses columns to give one value per **row**. Getting `axis` right is a daily task in ML code, so stare at these two outputs until they feel obvious.
""",
)

st.header("3 · Broadcasting: how a small array stretches to fit a big one")
st.markdown(
    """
What happens if you add arrays of *different* shapes? NumPy **broadcasts**:
it pretends to copy the smaller array along any dimension where its size
is 1 (or missing), so the shapes match — without actually using the memory.

**The rule:** line the two shapes up from the *right*. Each pair of
dimensions must be **equal**, or **one of them must be 1** (which stretches).
Otherwise: error.

Pick two shapes. Everything shown — values, stretched cells, or the error —
comes from NumPy really attempting `A + B`:
"""
)

SHAPES_A = {
    "(3, 4) matrix": np.array([[10, 20, 30, 40],
                               [50, 60, 70, 80],
                               [90, 100, 110, 120]]),
    "(3, 1) column": np.array([[10], [20], [30]]),
    "(1, 4) row": np.array([[10, 20, 30, 40]]),
}
SHAPES_B = {
    "(1, 4) row": np.array([[1, 2, 3, 4]]),
    "(3, 1) column": np.array([[1], [2], [3]]),
    "(4,) plain vector": np.array([1, 2, 3, 4]),
    "(3,) plain vector": np.array([1, 2, 3]),
    "scalar (just the number 7)": np.array(7),
}
ca, cb = st.columns(2)
name_a = ca.selectbox("shape of A", list(SHAPES_A.keys()))
name_b = cb.selectbox("shape of B", list(SHAPES_B.keys()), index=0)
A, B = SHAPES_A[name_a], SHAPES_B[name_b]


def draw_grid(ax, arr, virtual_mask, title, cmap_real="#bbdefb",
              cmap_virtual="#e3f2fd"):
    arr = np.atleast_2d(arr)
    virtual_mask = np.atleast_2d(virtual_mask)
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(cols):
            virt = virtual_mask[i, j]
            ax.add_patch(Rectangle((j, rows - 1 - i), 0.94, 0.94,
                                   facecolor=cmap_virtual if virt
                                   else cmap_real,
                                   edgecolor="#546e7a",
                                   linestyle=":" if virt else "-"))
            ax.text(j + 0.47, rows - 1 - i + 0.47, str(arr[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="#90a4ae" if virt else "#0d47a1",
                    style="italic" if virt else "normal")
    ax.set_xlim(-0.2, max(cols, 4) + 0.2)
    ax.set_ylim(-0.7, max(rows, 3) + 0.6)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


try:
    result = A + B  # the real broadcast
    out_shape = result.shape
    A_big = np.broadcast_to(A, out_shape)
    B_big = np.broadcast_to(B, out_shape)

    def virtual(orig, out_shape):
        """True where a cell is a broadcast copy, not original data."""
        o = np.atleast_2d(orig)
        mask = np.zeros(out_shape, dtype=bool)
        if o.shape[0] == 1 and out_shape[0] > 1:
            mask[1:, :] = True
        if o.shape[1] == 1 and out_shape[1] > 1:
            mask[:, 1:] = True
        if orig.ndim == 0:
            mask[:, :] = True
            mask[0, 0] = False
        return mask

    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    draw_grid(axes[0], A_big, virtual(A, out_shape),
              f"A  {A.shape} stretched to {out_shape}\n(dotted italics = "
              "virtual copies)")
    draw_grid(axes[1], B_big, virtual(B, out_shape),
              f"B  {B.shape} stretched to {out_shape}",
              cmap_real="#ffe0b2", cmap_virtual="#fff3e0")
    draw_grid(axes[2], result, np.zeros(out_shape, dtype=bool),
              f"A + B  →  {out_shape}", cmap_real="#c8e6c9")
    st.pyplot(fig)
    plt.close(fig)
    st.success(
        f"`{A.shape} + {B.shape}` broadcasts to `{out_shape}` — solid cells "
        "are real data, dotted cells are the copies NumPy *pretends* to make."
    )
except ValueError as exc:
    st.error(
        f"NumPy itself refused: `ValueError: {exc}`\n\n"
        f"Align the shapes from the right: `{A.shape}` vs `{B.shape}`. "
        "Some pair of dimensions is neither equal nor 1, so there is no "
        "legal way to stretch — this combination simply doesn't broadcast."
    )

st.caption(
    "Where you'll meet this for real: standardizing a dataset. "
    "`(1000, 5) - (5,)` broadcasts the 5 column means over all 1000 rows — "
    "that's `X - X.mean(axis=0)` working."
)

st.header("4 · Slicing a matrix: `arr[rows, cols]`")
rows_sel = st.slider("row slice  `r0:r1`", 0, 5, (1, 4))
cols_sel = st.slider("column slice  `c0:c1`", 0, 6, (2, 5))
M = np.arange(10, 10 + 30).reshape(5, 6)
sliced = M[rows_sel[0]:rows_sel[1], cols_sel[0]:cols_sel[1]]  # the real slice

fig, ax = plt.subplots(figsize=(6.5, 2.9))
for i in range(5):
    for j in range(6):
        inside = (rows_sel[0] <= i < rows_sel[1]
                  and cols_sel[0] <= j < cols_sel[1])
        ax.add_patch(Rectangle((j, 4 - i), 0.94, 0.94,
                               facecolor="#ffd54f" if inside else "#eceff1",
                               edgecolor="#546e7a"))
        ax.text(j + 0.47, 4 - i + 0.47, str(M[i, j]), ha="center",
                va="center", fontsize=9)
ax.set_title(f"M[{rows_sel[0]}:{rows_sel[1]}, {cols_sel[0]}:{cols_sel[1]}]  "
             f"→ shape {sliced.shape}", fontsize=10)
ax.set_xlim(-0.2, 6.2)
ax.set_ylim(-0.2, 5.2)
ax.set_aspect("equal")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
st.code(repr(sliced), language="text")
st.caption("Same rule as list slices: start included, stop excluded — "
           "and this really is `M[r0:r1, c0:c1]` evaluated live.")

sandbox(
    '''import numpy as np

# Five patients, three measurements: [age, weight_kg, blood_pressure]
data = np.array([[34, 70, 118],
                 [61, 82, 141],
                 [47, 65, 125],
                 [29, 90, 132],
                 [55, 77, 128]])

print("shape:", data.shape)
print("column means:", data.mean(axis=0))

# Standardize: subtract each column's mean (broadcasting in action!)
centered = data - data.mean(axis=0)
print("centered:\\n", centered.round(2))

# Boolean mask: patients with blood pressure over 130
high_bp = data[:, 2] > 130
print("high BP rows:\\n", data[high_bp])

# Challenge 1: also divide by data.std(axis=0) - full standardization
#              (this exact step is required by DeepSurv later!).
# Challenge 2: select only the ages (column 0) of high-BP patients.
# Challenge 3: try adding np.array([1, 2]) to data - read the error,
#              then explain to yourself why the shapes can't broadcast.''',
    key="p6",
)
