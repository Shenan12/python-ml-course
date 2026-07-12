import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import FancyArrowPatch, Rectangle

from utils.sandbox import sandbox, show_example

st.title("🏗️ Object-Oriented Programming")
st.markdown(
    """
So far your data (lists, dicts) and your behaviour (functions) live apart. A
**class** glues them together: it's a *blueprint* that says "everything of
this kind carries **this data** and can do **these actions**". Each concrete
thing built from the blueprint is an **object** (or *instance*).

Why you personally care: every ML tool you'll meet is a class.
`RandomForestClassifier()` builds an object; `model.fit(X, y)` calls one of
its **methods**; `model.n_estimators` reads one of its **attributes**. Learn
the pattern once here and every library page later reads itself.
"""
)

st.header("1 · One blueprint, many objects")
st.markdown(
    """
Below is a real class, and a little factory that builds real instances of it.
Add a dog or two — each box on the right is an actual Python object that was
just created with `Dog(name, age)`, and the attribute values shown are read
straight out of the live objects.
"""
)


class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        return f"{self.name} says woof!"


if "dogs" not in st.session_state:
    st.session_state.dogs = [Dog("Rex", 3), Dog("Bella", 5)]

with st.form("dog_form"):
    c1, c2, c3 = st.columns([2, 1, 1])
    new_name = c1.text_input("name", value="Milo")
    new_age = c2.number_input("age", 0, 25, 2)
    if c3.form_submit_button("Dog(name, age) →"):
        st.session_state.dogs.append(Dog(new_name, int(new_age)))
if len(st.session_state.dogs) > 4:
    st.session_state.dogs = st.session_state.dogs[-4:]

dogs = st.session_state.dogs
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.add_patch(Rectangle((0, 0.6), 2.7, 2.2, facecolor="#ede7f6",
                       edgecolor="#4527a0", linewidth=2))
ax.text(1.35, 2.55, "class Dog", ha="center", fontweight="bold", fontsize=11)
ax.text(0.2, 2.1, "data each dog carries:", fontsize=8, color="#4527a0")
ax.text(0.35, 1.8, "self.name\nself.age", fontsize=9, family="monospace")
ax.text(0.2, 1.25, "actions each dog can do:", fontsize=8, color="#4527a0")
ax.text(0.35, 0.95, "speak()", fontsize=9, family="monospace")

for i, dog in enumerate(dogs):
    x = 3.9 + (i % 2) * 2.9
    y = 1.75 - (i // 2) * 1.55
    ax.add_patch(Rectangle((x, y), 2.5, 1.25, facecolor="#e8f5e9",
                           edgecolor="#2e7d32"))
    # vars(dog) reads the REAL attribute dict of the live object:
    attrs = ", ".join(f"{k}={v!r}" for k, v in vars(dog).items())
    ax.text(x + 1.25, y + 0.95, f"a Dog object", ha="center", fontsize=9,
            fontweight="bold")
    ax.text(x + 1.25, y + 0.45, attrs, ha="center", fontsize=8,
            family="monospace")
    ax.add_patch(FancyArrowPatch((2.75, 1.7), (x - 0.05, y + 0.6),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#7e57c2",
                                 connectionstyle="arc3,rad=0.15"))
ax.text(3.2, 2.95, "each arrow = one call to Dog(...)", fontsize=8,
        color="#7e57c2")
ax.set_xlim(-0.2, 9.6)
ax.set_ylim(-0.1, 3.1)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

st.markdown("And calling `.speak()` on each **live object** right now gives:")
st.code("\n".join(dog.speak() for dog in dogs), language="text")

st.header("2 · The anatomy, line by line")
show_example(
    '''class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        return f"{self.name} says woof!"

    def birthday(self):
        self.age = self.age + 1

rex = Dog("Rex", 3)
print(rex.speak())
print("age before birthday:", rex.age)
rex.birthday()
print("age after birthday: ", rex.age)''',
    """
- `class Dog:` — starts a blueprint named `Dog`. By convention class names are `CapitalisedLikeThis`.
- `def __init__(self, name, age):` — the **initialiser**. Python runs it automatically every time you write `Dog(...)`. The double underscores mark it as a special, Python-recognised method.
- `self` — **the object currently being worked on**. When you call `rex.speak()`, Python silently rewrites it as `Dog.speak(rex)`, so inside the method `self` *is* `rex`. You never pass `self` yourself.
- `self.name = name` — takes the value that arrived in the parameter `name` and stores it **on the object**, as an attribute. Without `self.`, the value would vanish when `__init__` ends.
- `def speak(self):` — a method: an ordinary function that lives in the class and receives the object as `self`.
- `def birthday(self):` / `self.age = self.age + 1` — methods may **change** the object's attributes. This is why objects feel "alive": their state persists between calls.
- `rex = Dog("Rex", 3)` — builds an instance: Python creates an empty object, then calls `__init__` with `name="Rex"`, `age=3`.
- The two `print(... rex.age)` lines prove `birthday()` really modified the stored state.
""",
)

st.header("3 · Inheritance: a class built on top of another class")
st.markdown(
    """
**Inheritance** lets a new class say: *"I'm a `Dog`, plus extras."* The child
class gets everything the parent has for free, can **add** new methods, and
can **override** (replace) ones it wants to do differently.
"""
)
show_example(
    '''class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"

class Cat(Animal):                      # Cat inherits from Animal
    def speak(self):                    # OVERRIDE: replace the parent version
        return f"{self.name} says miaow"

class Kitten(Cat):                      # and Kitten inherits from Cat
    def pounce(self):                   # ADD: brand-new ability
        return f"{self.name} pounces!"

generic = Animal("Blob")
felix   = Cat("Felix")
mittens = Kitten("Mittens")

print(generic.speak())   # Animal's version
print(felix.speak())     # Cat's overriding version
print(mittens.speak())   # Kitten has no speak() -> Python finds Cat's
print(mittens.pounce())  # Kitten's own method''',
    """
- `class Cat(Animal):` — the brackets name the **parent**. `Cat` now automatically has `__init__` and `speak` without writing them.
- `def speak(self):` inside `Cat` — defining a method with the *same name* **overrides** the parent's version for all cats.
- `class Kitten(Cat):` — inheritance chains: `Kitten` → `Cat` → `Animal`.
- `mittens.speak()` — `Kitten` doesn't define `speak`, so Python *walks up the chain* and uses `Cat`'s. That walk is the mechanism visualised below.
- Note `Kitten` never wrote an `__init__` either — the one inherited from `Animal` (via `Cat`) ran when we built `mittens`.
""",
)

st.subheader("Watch Python walk the inheritance chain")
st.markdown(
    """
When you write `mittens.pounce()`, Python checks each class **in order** —
`Kitten`, then `Cat`, then `Animal` — and uses the *first* definition it
finds. Pick a method: the chain below is Python's real lookup order (its
`__mro__`), and the tick marks show where the method genuinely lives.
"""
)


class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} makes a sound"


class Cat(Animal):
    def speak(self):
        return f"{self.name} says miaow"


class Kitten(Cat):
    def pounce(self):
        return f"{self.name} pounces!"


method = st.selectbox("Call mittens.<method>() where mittens is a Kitten",
                      ["pounce", "speak", "__init__"])

mro = [cls for cls in Kitten.__mro__ if cls is not object]
defined_in = [cls.__name__ for cls in mro if method in cls.__dict__]
found_at = next(cls.__name__ for cls in mro if method in cls.__dict__)

fig, ax = plt.subplots(figsize=(9, 1.9))
x = 0.3
stopped = False
for cls in mro:
    name = cls.__name__
    has_it = method in cls.__dict__
    is_stop = name == found_at
    face = "#c8e6c9" if is_stop else ("#eceff1" if stopped else "#fff9c4")
    ax.add_patch(Rectangle((x, 0.55), 2.1, 0.95, facecolor=face,
                           edgecolor="#37474f"))
    ax.text(x + 1.05, 1.22, f"class {name}", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(x + 1.05, 0.85,
            (f"✓ defines {method}" if has_it else f"no {method} here"),
            ha="center", fontsize=8,
            color="#2e7d32" if has_it else "#90a4ae")
    if not stopped:
        ax.text(x + 1.05, 0.30,
                "FOUND — stop here" if is_stop else "checked… not found",
                ha="center", fontsize=8,
                color="#2e7d32" if is_stop else "#ef6c00")
    if is_stop:
        stopped = True
    if cls is not mro[-1]:
        ax.add_patch(FancyArrowPatch((x + 2.15, 1.0), (x + 2.85, 1.0),
                                     arrowstyle="-|>", mutation_scale=13,
                                     color="#546e7a"))
    x += 2.9
ax.text(0.3, 1.8, f"lookup order for  mittens.{method}  →", fontsize=9,
        color="#546e7a")
ax.set_xlim(0, 9.2)
ax.set_ylim(0.05, 2.0)
ax.axis("off")
st.pyplot(fig)
plt.close(fig)
st.success(
    f"Python finds `{method}` in **class {found_at}**"
    + (f" (also defined in {', '.join(defined_in[1:])}, but the earlier one "
       f"wins — that's overriding)" if len(defined_in) > 1 else "")
)

sandbox(
    '''# A tiny bank-account system. Run it, then try the challenges.
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return "Declined: insufficient funds"
        self.balance -= amount
        return self.balance

acct = BankAccount("Priya", 100)
print(acct.deposit(50))
print(acct.withdraw(30))
print(acct.withdraw(500))
print(f"{acct.owner} finishes with £{acct.balance}")

# Challenge 1: add an apply_interest(self, rate) method that multiplies
#              the balance by (1 + rate), and call it.
# Challenge 2: create a SavingsAccount(BankAccount) subclass that
#              overrides withdraw() to always decline. Prove it works.
# Challenge 3: create two separate accounts and show that depositing
#              into one does NOT change the other (objects are independent).''',
    key="p3",
)
