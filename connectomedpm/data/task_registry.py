"""The task suite.

DEVELOPMENT.md section 6 requires each parameter block to correspond to a *verifiable error
family* rather than a hand-labelled "capability". The registry therefore defines error
families, and every sample is generated from a named template so that:

* answers are exactly checkable (numeric or short string), and
* samples from one template form a group for splitting and clustered bootstrap.

Generation is a pure function of `seed`, so the suite can be rebuilt bit-for-bit.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Iterable

from ..utils.seed import numpy_rng
from .schemas import Sample

# Families reserved *entirely* for OOD evaluation. They never appear in D_block,
# D_router_train, D_router_dev, D_cal or D_test_ID.
OOD_TASK_FAMILIES: tuple[str, ...] = ("code_boundary", "symbolic_manipulation")

ALL_TASK_FAMILIES: tuple[str, ...] = (
    "numerical_arithmetic",
    "unit_conversion",
    "format_following",
    "structured_extraction",
    "logic_constraint",
    "factual_correction",
    "code_boundary",
    "symbolic_manipulation",
)

PROMPT_TEMPLATE = (
    "Answer with only the final value. Do not explain.\n"
    "Question: {question}\n"
    "Answer:"
)


def wrap(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def _fmt_number(x) -> str:
    f = float(x)
    if f.is_integer():
        return str(int(f))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def _stable_seed(*parts) -> int:
    """Seed that does not depend on python's randomised `hash`."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


_Template = tuple[str, Callable]


def _arith_templates() -> list[_Template]:
    def addition(i, rng):
        a = int(rng.integers(10_000, 99_999))
        b = int(rng.integers(10_000, 99_999))
        return f"What is {a} + {b}?", str(a + b)

    def subtraction(i, rng):
        a = int(rng.integers(100_000, 999_999))
        b = int(rng.integers(10_000, 99_999))
        return f"What is {a} - {b}?", str(a - b)

    def multiplication(i, rng):
        a = int(rng.integers(120, 989))
        b = int(rng.integers(120, 989))
        return f"What is {a} * {b}?", str(a * b)

    def percent(i, rng):
        pct = int(rng.choice([6, 8, 12, 14, 18, 22, 35, 45, 65, 85]))
        base = int(rng.choice([120, 250, 360, 480, 750, 1250, 2400, 3600]))
        return f"What is {pct} percent of {base}?", _fmt_number(pct * base / 100)

    def power(i, rng):
        base = int(rng.integers(2, 6))
        exp = int(rng.integers(7, 13))
        return f"What is {base} raised to the power {exp}?", str(base**exp)

    def gcd(i, rng):
        a = int(rng.integers(60, 4000))
        b = int(rng.integers(60, 4000))
        return f"What is gcd({a}, {b})?", str(math.gcd(a, b))

    def lcm(i, rng):
        a = int(rng.integers(12, 400))
        b = int(rng.integers(12, 400))
        return f"What is lcm({a}, {b})?", str(abs(a * b) // math.gcd(a, b))

    def mean(i, rng):
        start = int(rng.integers(20, 400))
        vals = [start, start + 6, start + 14, start + 28]
        q = f"What is the average of {vals[0]}, {vals[1]}, {vals[2]} and {vals[3]}?"
        return q, _fmt_number(sum(vals) / 4)

    def sqrt(i, rng):
        n = int(rng.integers(12, 140))
        return f"What is sqrt({n * n})?", str(n)

    return [
        ("addition", addition),
        ("subtraction", subtraction),
        ("multiplication", multiplication),
        ("percent", percent),
        ("power", power),
        ("gcd", gcd),
        ("lcm", lcm),
        ("mean", mean),
        ("sqrt", sqrt),
    ]


def _unit_templates() -> list[_Template]:
    def km_m(i, rng):
        v = int(rng.integers(3, 900))
        return f"How many meters are in {v} kilometers?", str(v * 1000)

    def kg_g(i, rng):
        v = int(rng.integers(2, 400))
        return f"How many grams are in {v} kilograms?", str(v * 1000)

    def h_min(i, rng):
        v = int(rng.integers(3, 90))
        return f"How many minutes are in {v} hours?", str(v * 60)

    def c_f(i, rng):
        v = int(rng.integers(-20, 120))
        return f"Convert {v} degrees Celsius to degrees Fahrenheit.", _fmt_number(v * 9 / 5 + 32)

    def min_s(i, rng):
        v = int(rng.integers(4, 120))
        return f"How many seconds are in {v} minutes?", str(v * 60)

    def mb_b(i, rng):
        v = int(rng.integers(2, 64))
        return (f"How many bytes are in {v} megabytes, using 1 megabyte = 1048576 bytes?",
                str(v * 1048576))

    return [
        ("km_to_m", km_m),
        ("kg_to_g", kg_g),
        ("hours_to_minutes", h_min),
        ("celsius_to_fahrenheit", c_f),
        ("minutes_to_seconds", min_s),
        ("megabyte_to_byte", mb_b),
    ]


def _format_templates() -> list[_Template]:
    def parity_word(i, rng):
        n = int(rng.integers(1000, 99999))
        return f"Is {n} even or odd? Reply with exactly one word: EVEN or ODD.", ("EVEN" if n % 2 == 0 else "ODD")

    def upper_word(i, rng):
        words = ["orbit", "lantern", "quartz", "meadow", "cobalt", "tundra", "vessel", "garnet"]
        w = str(rng.choice(words))
        return f"Write the word '{w}' in uppercase. Reply with only that word.", w.upper()

    def yes_no(i, rng):
        a = int(rng.integers(3, 400))
        b = int(rng.integers(3, 400))
        return f"Reply with YES or NO only: is {a} greater than {b}?", ("YES" if a > b else "NO")

    def digits(i, rng):
        n = int(rng.integers(10000, 999999))
        return f"How many digits does the number {n} have? Reply with a single number.", str(len(str(n)))

    def reverse(i, rng):
        n = int(rng.integers(1000, 99999))
        return (f"Write the digits of {n} in reverse order. Reply with only the resulting number.",
                str(n)[::-1])

    return [
        ("parity_word", parity_word),
        ("uppercase", upper_word),
        ("yes_no_compare", yes_no),
        ("count_digits", digits),
        ("reverse_digits", reverse),
    ]


def _extraction_templates() -> list[_Template]:
    def json_field(i, rng):
        value = int(rng.integers(100, 9999))
        keys = ["count", "total", "score", "weight", "size"]
        key = str(rng.choice(keys))
        others = ", ".join(f'"{k}": {int(rng.integers(1, 99))}' for k in keys if k != key)
        question = f'In the JSON object {{{others}, "{key}": {value}}}, what is the value of "{key}"?'
        return question, str(value)

    def key_value(i, rng):
        value = int(rng.integers(10, 999))
        return f"In the line 'id=7; port={value}; retries=3', what is the value of port?", str(value)

    def csv_column(i, rng):
        value = int(rng.integers(10, 999))
        return f"Given the CSV row 'alpha,{value},gamma', what is the second field?", str(value)

    return [
        ("json_field", json_field),
        ("key_value_parse", key_value),
        ("csv_second_field", csv_column),
    ]


def _logic_templates() -> list[_Template]:
    def removal(i, rng):
        total = int(rng.integers(20, 200))
        given = int(rng.integers(3, 18))
        return f"A box has {total} items. {given} are removed. How many remain?", str(total - given)

    def age_future(i, rng):
        now = int(rng.integers(20, 60))
        delta = int(rng.integers(3, 20))
        return f"Ada is {now} years old. How old will she be in {delta} years?", str(now + delta)

    def speed_distance(i, rng):
        speed = int(rng.choice([40, 50, 60, 70, 80, 90, 100]))
        hours = int(rng.integers(2, 9))
        return (f"A train travels at {speed} km/h for {hours} hours. How many km does it travel?",
                str(speed * hours))

    def equal_share(i, rng):
        each = int(rng.integers(3, 40))
        people = int(rng.integers(3, 25))
        return (f"{people} people share {each * people} sweets equally. How many does each get?",
                str(each))

    return [
        ("removal", removal),
        ("age_future", age_future),
        ("speed_distance", speed_distance),
        ("equal_share", equal_share),
    ]


_FACTS: list[tuple[str, str]] = [
    ("How many bits are in one byte?", "8"),
    ("How many grams are in one kilogram?", "1000"),
    ("How many meters are in one kilometer?", "1000"),
    ("How many degrees are in a right angle?", "90"),
    ("How many degrees are in a full circle?", "360"),
    ("What is the SI unit symbol for electric current?", "A"),
    ("What is the SI unit symbol for force?", "N"),
    ("What is the SI unit symbol for energy?", "J"),
    ("What is the chemical symbol for potassium?", "K"),
    ("What is the chemical symbol for silver?", "Ag"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("How many sides does a hexagon have?", "6"),
    ("How many sides does a pentagon have?", "5"),
    ("How many minutes are in one hour?", "60"),
    ("How many seconds are in one minute?", "60"),
    ("What is the boiling point of water in degrees Celsius at sea level?", "100"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("How many players are on a football (soccer) team on the field?", "11"),
    ("How many strings does a standard guitar have?", "6"),
    ("How many continents are there on Earth?", "7"),
]


def _factual_templates() -> list[_Template]:
    def make(fact_idx: int):
        question, answer = _FACTS[fact_idx]

        def _t(i, rng):
            return question, answer

        return _t

    return [(f"fact_{idx:02d}", make(idx)) for idx in range(len(_FACTS))]


def _code_templates() -> list[_Template]:
    def range_len(i, rng):
        a = int(rng.integers(0, 40))
        b = int(rng.integers(a + 1, a + 60))
        return f"In Python, what does len(range({a}, {b})) return?", str(b - a)

    def last_index(i, rng):
        n = int(rng.integers(2, 80))
        return f"In Python, for a list of length {n}, what is the index of the last element?", str(n - 1)

    def slice_length(i, rng):
        a = int(rng.integers(0, 30))
        b = int(rng.integers(a + 2, a + 40))
        return f"In Python, if s has length 100, what is the length of s[{a}:{b}]?", str(b - a)

    def floor_division(i, rng):
        a = int(rng.integers(20, 999))
        b = int(rng.integers(2, 20))
        return f"In Python, what does {a} // {b} evaluate to?", str(a // b)

    def modulo(i, rng):
        a = int(rng.integers(20, 999))
        b = int(rng.integers(2, 20))
        return f"In Python, what does {a} % {b} evaluate to?", str(a % b)

    return [
        ("range_length", range_len),
        ("last_index", last_index),
        ("slice_length", slice_length),
        ("floor_division", floor_division),
        ("modulo", modulo),
    ]


def _symbolic_templates() -> list[_Template]:
    def affine_eval(i, rng):
        a = int(rng.integers(2, 12))
        b = int(rng.integers(-15, 25))
        x = int(rng.integers(-9, 15))
        return f"If f(x) = {a}x + {b}, what is f({x})?", str(a * x + b)

    def compose_eval(i, rng):
        a = int(rng.integers(2, 9))
        x = int(rng.integers(1, 12))
        return f"If g(x) = x + {a}, and h(x) = 3x, what is h(g({x}))?", str(3 * (x + a))

    def constant_term(i, rng):
        a = int(rng.integers(2, 12))
        return (f"If p(x) = x + {a} and q(x) = x, what is p(0)*q(0)?", "0")

    def quadratic_eval(i, rng):
        x = int(rng.integers(2, 12))
        a = int(rng.integers(1, 5))
        return f"If f(x) = {a}x^2 - x, what is f({x})?", str(a * x * x - x)

    return [
        ("affine_eval", affine_eval),
        ("compose_eval", compose_eval),
        ("constant_term", constant_term),
        ("quadratic_eval", quadratic_eval),
    ]


_FAMILY_BUILDERS: dict[str, Callable[[], list[_Template]]] = {
    "numerical_arithmetic": _arith_templates,
    "unit_conversion": _unit_templates,
    "format_following": _format_templates,
    "structured_extraction": _extraction_templates,
    "logic_constraint": _logic_templates,
    "factual_correction": _factual_templates,
    "code_boundary": _code_templates,
    "symbolic_manipulation": _symbolic_templates,
}


def build_task_suite(
    seed: int = 0,
    per_template: int = 12,
    families: Iterable[str] | None = None,
    family_overrides: dict | None = None,
) -> list[Sample]:
    """Deterministically generate the task suite.

    Each template gets its own RNG stream, so changing one family's size cannot perturb
    another family's samples.
    """
    families = list(families or ALL_TASK_FAMILIES)
    family_overrides = family_overrides or {}
    samples: list[Sample] = []

    for family in families:
        builder = _FAMILY_BUILDERS[family]
        count = int(family_overrides.get(family, per_template))
        for tname, fn in builder():
            template_family = f"{family}::{tname}"
            rng = numpy_rng(_stable_seed(seed, family, tname))
            for i in range(count):
                question, answer = fn(i, rng)
                samples.append(
                    Sample(
                        sample_id=f"{family}:{tname}:{i:04d}",
                        prompt=wrap(question),
                        answer=answer,
                        task_family=family,
                        template_family=template_family,
                        metadata={"template": tname, "question": question},
                    )
                )
    samples.sort(key=lambda s: s.sample_id)
    return samples
