"""Build sample_sales.csv: a realistic, deliberately messy sales dataset for demos and tests.

Usage:
    python sample_data/make_sample_data.py              # writes sample_data/sample_sales.csv
    python sample_data/make_sample_data.py --output x.csv

The data is random but reproducible: the same seed always gives the same file,
byte for byte. It's synthetic: no real customers or orders.

Planted problems (the answer key; the EDA should find every one of them)
-------------------------------------------------------------------------
- order_date        dates stored as text, day first (31/12/2025)
- region            inconsistent labels ("north") and trailing spaces ("South ")
- discount_pct      ~12% missing, mostly for Store orders (missing at random, MAR)
- delivery_days     missing for every Store order (not applicable, ~30%), and the
                    code 999 for "unknown" in a few rows: a placeholder the
                    skill can't recognise, so it shows up as an outlier instead
- shipping_cost     numbers stored as text ("$4.99"), plus the word "free"
- rating            numbers stored as text, with "-" and "n/a" for "no rating".
                    pandas turns "n/a" into a real missing value while reading
                    the CSV; "-" survives as text and is reported as a placeholder
- currency          constant: always "USD"
- legacy_code       completely empty
- quantity, revenue a few wholesale orders (quantity 150-300): real but extreme
- unit_price, revenue  highly right-skewed, as prices and order values usually are
- revenue           strongly correlated with unit_price (it is computed from it)
- (rows)            12 exact duplicate rows, and 3 order_ids that appear twice
                    with a different quantity (run the EDA with --key order_id)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2025
N_ORDERS = 1200
N_CUSTOMERS = 350
YEAR = 2025

CATEGORIES = {  # median unit price
    "Electronics": 120.0,
    "Home": 35.0,
    "Garden": 28.0,
    "Toys": 18.0,
    "Books": 12.0,
    "Sports": 45.0,
}
REGIONS = {"North": 0.30, "South": 0.25, "East": 0.20, "West": 0.25}
CHANNELS = {"Online": 0.55, "Store": 0.30, "Phone": 0.15}

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sample_sales.csv"


def make_sales(seed: int = SEED, n: int = N_ORDERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Orders through the year, busier in November and December.
    day_weights = np.ones(365)
    day_weights[304:] = 1.8
    days = np.sort(rng.choice(365, size=n, p=day_weights / day_weights.sum()))
    dates = pd.Timestamp(f"{YEAR}-01-01") + pd.to_timedelta(days, unit="D")

    # A few customers order a lot (roughly Pareto), most only once or twice.
    weights = 1 / np.arange(1, N_CUSTOMERS + 1) ** 0.8
    customers = rng.choice(N_CUSTOMERS, size=n, p=weights / weights.sum()) + 1

    region = rng.choice(list(REGIONS), size=n, p=list(REGIONS.values())).astype(object)
    channel = rng.choice(list(CHANNELS), size=n, p=list(CHANNELS.values()))
    category = rng.choice(list(CATEGORIES), size=n)

    quantity = np.minimum(rng.geometric(0.45, size=n), 10)
    medians = np.array([CATEGORIES[c] for c in category])
    unit_price = np.round(medians * rng.lognormal(0, 0.6, size=n), 2)

    discount = rng.choice([0, 5, 10, 15, 20], size=n, p=[0.45, 0.2, 0.2, 0.1, 0.05]).astype(float)
    is_store = channel == "Store"
    gap_chance = np.where(is_store, 0.30, 0.05)  # MAR: gaps depend on the channel
    discount[rng.random(n) < gap_chance] = np.nan

    delivery = rng.integers(1, 8, size=n).astype(float)
    delivery[is_store] = np.nan  # structural: store orders aren't delivered

    shipping = np.where(is_store, 0.0, rng.choice([4.99, 7.99, 12.99], size=n))
    shipping_text = np.array([f"${v:.2f}" for v in shipping], dtype=object)

    rating = rng.choice(["1", "2", "3", "4", "5"], size=n, p=[0.06, 0.09, 0.2, 0.35, 0.3]).astype(object)

    df = pd.DataFrame(
        {
            "order_id": np.arange(100001, 100001 + n),
            "order_date": dates.strftime("%d/%m/%Y"),
            "customer_id": [f"C{c:05d}" for c in customers],
            "region": region,
            "channel": channel,
            "category": category,
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": discount,
            "revenue": 0.0,  # filled in after the planted problems below
            "shipping_cost": shipping_text,
            "delivery_days": delivery,
            "rating": rating,
            "returning_customer": rng.random(n) < 0.4,
            "currency": "USD",
            "legacy_code": np.nan,
        }
    )

    # --- planted problems ----------------------------------------------------
    rows = rng.permutation(n)  # disjoint random rows for each problem
    sizes = {"north": 24, "south_space": 12, "wholesale": 4, "no_rating": 90, "na_rating": 40}
    groups = dict(zip(sizes, np.split(rows, np.cumsum(list(sizes.values())))))

    df.loc[groups["north"], "region"] = "north"  # inconsistent case
    df.loc[groups["south_space"], "region"] = "South "  # trailing space
    df.loc[groups["wholesale"], "quantity"] = rng.integers(150, 301, size=sizes["wholesale"])
    df.loc[groups["no_rating"], "rating"] = "-"
    df.loc[groups["na_rating"], "rating"] = "n/a"
    online = df.index[df["channel"] != "Store"]
    df.loc[rng.choice(online, 3, replace=False), "delivery_days"] = 999  # "unknown" code
    df.loc[rng.choice(online, 5, replace=False), "shipping_cost"] = "free"

    df["revenue"] = np.round(df["quantity"] * df["unit_price"] * (1 - df["discount_pct"].fillna(0) / 100), 2)

    # Exact duplicates (e.g. an export that ran twice) and repeated keys with
    # conflicting values (e.g. an order edited after the first export).
    exact = df.iloc[np.sort(rng.choice(n, 12, replace=False))]
    edited = df.iloc[np.sort(rng.choice(n, 3, replace=False))].copy()
    edited["quantity"] = edited["quantity"] + 1
    edited["revenue"] = np.round(
        edited["quantity"] * edited["unit_price"] * (1 - edited["discount_pct"].fillna(0) / 100), 2
    )
    df = pd.concat([df, exact, edited])
    return df.sort_values("order_id", kind="stable").reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "\n" line endings and a fixed float format keep the file identical on every OS.
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.2f")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the messy sample sales dataset.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"CSV path (default {DEFAULT_OUTPUT.name})")
    p.add_argument("--seed", type=int, default=SEED, help=f"random seed (default {SEED})")
    args = p.parse_args(argv)
    df = make_sales(seed=args.seed)
    path = write_csv(df, args.output)
    print(f"Wrote {path} ({len(df):,} rows x {df.shape[1]} columns, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
