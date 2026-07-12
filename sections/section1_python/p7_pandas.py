import pandas as pd
import streamlit as st

from utils.sandbox import sandbox, show_example

st.title("🐼 Pandas: DataFrames")
st.markdown(
    """
NumPy gave you a grid of numbers. **Pandas** wraps that grid into a
**DataFrame** — a table with *named columns* and *labelled rows*, like a
spreadsheet that answers to code. Loading a CSV, cleaning it, filtering rows,
summarising by group, joining two tables: that's 80% of every real data job,
and it all happens in pandas. (A single column of a DataFrame is called a
**Series**.)
"""
)

students = pd.DataFrame({
    "name": ["Priya", "Tom", "Aisha", "Ben", "Zara", "Leo", "Maya", "Sam",
             "Ivy"],
    "house": ["Newton", "Curie", "Newton", "Turing", "Curie", "Turing",
              "Newton", "Curie", "Turing"],
    "mark": [68, 91, 75, 55, 82, 61, 88, 73, 70],
    "attendance": [0.92, 0.98, 0.85, 0.70, 0.95, 0.80, 0.99, 0.88, 0.91],
})

st.header("1 · Anatomy: index, columns, values")
show_example(
    '''import pandas as pd

students = pd.DataFrame({
    "name": ["Priya", "Tom", "Aisha", "Ben"],
    "house": ["Newton", "Curie", "Newton", "Turing"],
    "mark": [68, 91, 75, 55],
})

print(students)
print()
print("columns:", list(students.columns))
print("index:  ", list(students.index))
print("shape:  ", students.shape)
print()
print(students["mark"].describe().round(2))''',
    """
- `pd.DataFrame({...})` — builds a table from a dict: each **key becomes a column name**, each list becomes that column's values.
- The printed table — the unnamed left-hand column of `0 1 2 3` is the **index**: a label for each row. By default it's just the row number.
- `students.columns` / `students.index` / `students.shape` — the three pieces of anatomy: column names, row labels, and (rows, columns) — the same `.shape` idea as NumPy, because a DataFrame is a NumPy array underneath.
- `students["mark"]` — one column, pulled out by name. This is a **Series**.
- `.describe()` — instant summary statistics of a Series (count, mean, std, quartiles). `.round(2)` just tidies the decimals.
""",
)

st.header("2 · Picking rows and columns: `loc`, `iloc` and masks")
show_example(
    '''import pandas as pd

students = pd.DataFrame({
    "name": ["Priya", "Tom", "Aisha", "Ben"],
    "house": ["Newton", "Curie", "Newton", "Turing"],
    "mark": [68, 91, 75, 55],
})

print(students.loc[2, "name"])        # loc = BY LABEL (row label 2)
print(students.iloc[0, 2])            # iloc = BY POSITION (row 0, col 2)
print()

high = students[students["mark"] >= 70]     # boolean mask, as in NumPy
print(high)
print()

newton_marks = students.loc[students["house"] == "Newton", "mark"]
print("Newton marks:", list(newton_marks), "mean:", newton_marks.mean())''',
    """
- `students.loc[2, "name"]` — `.loc` selects **by label**: the row whose index label is `2`, column `"name"`.
- `students.iloc[0, 2]` — `.iloc` selects **by integer position**, exactly like NumPy `M[0, 2]`. The distinction matters once indexes stop being 0,1,2,… (e.g. after filtering or when dates are the index).
- `students["mark"] >= 70` — a boolean mask Series (this is NumPy's mask idea wearing a pandas coat).
- `students[mask]` — keeps only the `True` rows. **This is the single most-used pattern in data work.**
- `students.loc[mask, "mark"]` — mask for rows *and* a column name at once: "the marks of Newton students". Then `.mean()` on the result.
""",
)

st.header("3 · GroupBy: split → apply → combine, watched live")
st.markdown(
    """
`groupby` is the most powerful idea on this page. One line —
`students.groupby("house")["mark"].mean()` — secretly performs three steps:

1. **SPLIT** the table into one mini-table per house
2. **APPLY** a calculation (mean, max, count…) to each mini-table
3. **COMBINE** the answers into one small result

Every table below is the real thing: pandas genuinely splits this DataFrame
(via `groupby.get_group`), applies your chosen function, and combines.
"""
)

agg_name = st.radio("Step 2 — the function to APPLY to each group's marks",
                    ["mean", "max", "count"], horizontal=True)

HOUSE_COLOURS = {"Newton": "#e3f2fd", "Curie": "#fce4ec", "Turing": "#e8f5e9"}


def paint(df):
    return df.style.apply(
        lambda row: [f"background-color: {HOUSE_COLOURS[row['house']]}"]
        * len(row), axis=1)


st.markdown("**The original table** (already coloured by house):")
st.dataframe(paint(students), hide_index=True, width="stretch")

gb = students.groupby("house")
st.markdown("**Step 1 — SPLIT** into one mini-table per house:")
cols = st.columns(len(gb.groups))
for col, (house, _) in zip(cols, gb.groups.items()):
    with col:
        st.dataframe(paint(gb.get_group(house)), hide_index=True,
                     width="stretch")

st.markdown(f"**Step 2 — APPLY** `{agg_name}` to each mini-table's "
            "`mark` column:")
applied = gb["mark"].agg(agg_name)  # the real computation
cols = st.columns(len(applied))
for col, (house, value) in zip(cols, applied.items()):
    col.metric(f"{house}", f"{value:.2f}" if agg_name == "mean" else value)

st.markdown("**Step 3 — COMBINE** into the final result — which is exactly "
            f"what the one-liner returns:")
st.code(f'students.groupby("house")["mark"].{agg_name}()\n\n'
        + str(applied), language="text")

st.header("4 · Merging: joining two tables on a shared column")
st.markdown(
    """
Real data arrives in pieces: patient details in one table, lab results in
another. `pd.merge` stitches them together by matching values in a shared
**key column** — here, `house`. Note the mismatch built into the tables
below: the students include **Turing** house (missing from the staff table),
and the staff table includes **Lovelace** (which has no students). Choose a
join type and watch who survives:
"""
)

staff = pd.DataFrame({
    "house": ["Newton", "Curie", "Lovelace"],
    "head_of_house": ["Dr Patel", "Dr Chen", "Dr Okoro"],
})
c1, c2 = st.columns(2)
c1.markdown("**Left table — one row per student:**")
c1.dataframe(students[["name", "house"]], hide_index=True, width="stretch")
c2.markdown("**Right table — one row per house:**")
c2.dataframe(staff, hide_index=True, width="stretch")

how = st.radio(
    "Join type (`how=`)",
    ["inner — keep only matching houses", "left — keep every student",
     "right — keep every staff row", "outer — keep absolutely everything"],
    horizontal=True,
).split(" ")[0]

merged = pd.merge(students[["name", "house"]], staff, on="house", how=how,
                  indicator=True)  # indicator adds a column saying WHY
SOURCE_COLOURS = {"both": "#c8e6c9", "left_only": "#ffecb3",
                  "right_only": "#ffcdd2"}
styled = merged.style.apply(
    lambda row: [f"background-color: {SOURCE_COLOURS[row['_merge']]}"]
    * len(row), axis=1)
st.dataframe(styled, hide_index=True, width="stretch")
counts = merged["_merge"].value_counts()
st.markdown(
    f"""
`pd.merge(..., how="{how}", indicator=True)` produced **{len(merged)} rows**
(🟢 both tables: {counts.get('both', 0)} · 🟡 left only:
{counts.get('left_only', 0)} · 🔴 right only: {counts.get('right_only', 0)}).
The `_merge` column is pandas itself reporting where each row came from —
notice how missing partners appear as `NaN` (pandas' "no value here" marker).
"""
)

show_example(
    '''import pandas as pd

students = pd.DataFrame({"name": ["Priya", "Tom", "Ben"],
                         "house": ["Newton", "Curie", "Turing"]})
staff = pd.DataFrame({"house": ["Newton", "Curie", "Lovelace"],
                      "head_of_house": ["Dr Patel", "Dr Chen", "Dr Okoro"]})

merged = pd.merge(students, staff, on="house", how="left")
print(merged)
print()
print("rows with a missing head:")
print(merged[merged["head_of_house"].isna()])''',
    """
- `pd.merge(students, staff, on="house", how="left")` — match rows where the `house` values are equal. `how="left"` promises: *every* row of the left table survives, matched or not.
- Ben's row — Turing has no staff entry, so his `head_of_house` is `NaN` ("Not a Number", pandas' missing-value marker).
- `.isna()` — a mask that is `True` where values are missing; filtering with it is how you *audit* a merge. **Always check for unexpected `NaN`s after merging** — silent key mismatches are a classic real-world data bug.
""",
)

sandbox(
    '''import pandas as pd

sales = pd.DataFrame({
    "shop":    ["North", "North", "South", "South", "East", "East", "East"],
    "product": ["tea", "coffee", "tea", "coffee", "tea", "coffee", "juice"],
    "units":   [12, 30, 8, 25, 15, 22, 5],
    "price":   [2.5, 3.5, 2.5, 3.5, 2.5, 3.5, 4.0],
})
sales["revenue"] = sales["units"] * sales["price"]
print(sales)
print()
print(sales.groupby("shop")["revenue"].sum())

# Challenge 1: group by "product" instead - which product earns most?
# Challenge 2: filter to rows with units > 10 BEFORE grouping.
# Challenge 3: merge in this second table with how="left", then find
#              which shop has a missing manager:
managers = pd.DataFrame({"shop": ["North", "South"],
                         "manager": ["Ana", "Raj"]})''',
    key="p7",
)
