"""Generate a realistic, reproducible sales database for QuerentCrew.

The data contains a planted business story so the demo has something to find:
revenue in the South region drops in Q3 2025, driven mainly by the Electronics
category and by Enterprise customers ordering less.

Usage:
    python scripts/seed_database.py                 # writes data/sales.db
    python scripts/seed_database.py --path other.db --seed 7
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["North", "South", "East", "West"]
SEGMENTS = ["Consumer", "SMB", "Enterprise"]
CHANNELS = ["Online", "Retail", "Partner"]

PRODUCTS = [
    # name, category, unit_price
    ("Laptop Pro 14", "Electronics", 1200.0),
    ("Laptop Air 13", "Electronics", 950.0),
    ("4K Monitor 27", "Electronics", 420.0),
    ("Wireless Earbuds", "Electronics", 120.0),
    ("Smartphone X", "Electronics", 800.0),
    ("Office Chair Ergo", "Furniture", 310.0),
    ("Standing Desk", "Furniture", 560.0),
    ("Bookshelf Oak", "Furniture", 180.0),
    ("Printer Paper (Box)", "Office Supplies", 35.0),
    ("Gel Pens (Pack)", "Office Supplies", 12.0),
    ("Notebook Set", "Office Supplies", 18.0),
    ("Desk Organizer", "Office Supplies", 25.0),
    ("Cloud Backup (1 yr)", "Software", 99.0),
    ("Antivirus Suite", "Software", 59.0),
    ("Design Studio License", "Software", 349.0),
]

FIRST = ["Aarav", "Diya", "Rohan", "Isha", "Kabir", "Meera", "Arjun", "Sara", "Vikram", "Anaya",
         "Liam", "Emma", "Noah", "Olivia", "Ethan", "Ava", "Mason", "Mia", "Lucas", "Zara"]
LAST = ["Sharma", "Patel", "Iyer", "Khan", "Reddy", "Gupta", "Singh", "Das", "Smith", "Brown",
        "Wilson", "Taylor", "Clark", "Lewis", "Walker", "Hall", "Young", "King", "Wright", "Green"]


def build_frames(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    regions = pd.DataFrame({"region_id": range(1, len(REGIONS) + 1), "name": REGIONS})

    products = pd.DataFrame(PRODUCTS, columns=["name", "category", "unit_price"])
    products.insert(0, "product_id", range(1, len(products) + 1))

    n_customers = 400
    first = rng.choice(FIRST, n_customers)
    last = rng.choice(LAST, n_customers)
    customers = pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "name": [f"{f} {s}" for f, s in zip(first, last)],
        "email": [f"{f.lower()}.{s.lower()}{i}@example.com" for i, (f, s) in enumerate(zip(first, last), 1)],
        "phone": [f"+91-9{rng.integers(100000000, 999999999)}" for _ in range(n_customers)],
        "segment": rng.choice(SEGMENTS, n_customers, p=[0.55, 0.30, 0.15]),
        "region_id": rng.integers(1, len(REGIONS) + 1, n_customers),
        "signup_date": pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 700, n_customers), unit="D"),
    })
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")

    # Orders across 2025
    days = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    order_rows, item_rows = [], []
    order_id, item_id = 1, 1
    cust_region = dict(zip(customers.customer_id, customers.region_id))
    cust_segment = dict(zip(customers.customer_id, customers.segment))
    south_id = REGIONS.index("South") + 1

    for day in days:
        n_orders = rng.poisson(9)
        for _ in range(n_orders):
            cid = int(rng.integers(1, n_customers + 1))
            region, segment = cust_region[cid], cust_segment[cid]
            in_q3 = day.month in (7, 8, 9)
            # Planted story: South + Q3 -> Enterprise customers order much less
            if in_q3 and region == south_id and segment == "Enterprise" and rng.random() < 0.7:
                continue
            status = rng.choice(["completed", "returned", "cancelled"], p=[0.92, 0.05, 0.03])
            order_rows.append({
                "order_id": order_id,
                "customer_id": cid,
                "order_date": day.strftime("%Y-%m-%d"),
                "status": status,
                "channel": rng.choice(CHANNELS, p=[0.5, 0.35, 0.15]),
            })
            for _ in range(int(rng.integers(1, 4))):
                pid = int(rng.integers(1, len(products) + 1))
                category = products.loc[pid - 1, "category"]
                # Planted story: South + Q3 -> Electronics demand drops
                if in_q3 and region == south_id and category == "Electronics" and rng.random() < 0.55:
                    pid = int(rng.choice(products.index[products.category == "Office Supplies"]) + 1)
                base_price = float(products.loc[pid - 1, "unit_price"])
                qty = int(rng.integers(1, 6 if segment != "Enterprise" else 15))
                discount = float(rng.choice([0, 0, 0, 0.05, 0.10, 0.15]))
                item_rows.append({
                    "order_item_id": item_id,
                    "order_id": order_id,
                    "product_id": pid,
                    "quantity": qty,
                    "unit_price": base_price,
                    "discount": discount,
                })
                item_id += 1
            order_id += 1

    return {
        "regions": regions,
        "products": products,
        "customers": customers,
        "orders": pd.DataFrame(order_rows),
        "order_items": pd.DataFrame(item_rows),
    }


SCHEMA_SQL = """
CREATE TABLE regions (
    region_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    segment TEXT NOT NULL,
    region_id INTEGER NOT NULL REFERENCES regions(region_id),
    signup_date TEXT
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL,
    channel TEXT NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount REAL NOT NULL DEFAULT 0
);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
"""


def seed(path: str | Path = "data/sales.db", seed_value: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    frames = build_frames(seed_value)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        for table in ["regions", "products", "customers", "orders", "order_items"]:
            frames[table].to_sql(table, conn, if_exists="append", index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="data/sales.db")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = seed(args.path, args.seed)
    with sqlite3.connect(out) as conn:
        for t in ["regions", "products", "customers", "orders", "order_items"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:<12} {n:>6} rows")
    print(f"Database written to {out}")


if __name__ == "__main__":
    main()
