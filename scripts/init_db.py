"""
One-time script to initialise the PostgreSQL database.

What it does:
  1. Creates the `predictions` table (empty — the API writes to this at runtime)
  2. Loads diabetes_processed.csv into the `features` table

Run once after `docker compose up`:
    python3 scripts/init_db.py

Safe to re-run — `predictions` table is not touched,
`features` table is replaced with fresh data from the CSV.
"""

import sys
from pathlib import Path

# allow imports from project root (src/, configs/) — must come before src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.db import Base, get_engine


PROCESSED_CSV = Path(__file__).resolve().parent.parent / "data" / "diabetes_processed.csv"


def init_db():
    print("Connecting to database...")
    engine = get_engine()

    # create all ORM-defined tables (predictions, etc.)
    print("Creating tables...")
    Base.metadata.create_all(engine)
    print("predictions table ready")

    # load processed dataset into features table
    if not PROCESSED_CSV.exists():
        print(f"CSV not found at {PROCESSED_CSV}")
        print("    Run: python3 -c \"from src.data import build_processed_data; build_processed_data()\"")
        sys.exit(1)

    print(f"Loading {PROCESSED_CSV.name} into features table...")
    df = pd.read_csv(PROCESSED_CSV)
    print(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

    df.to_sql(
        name="features",
        con=engine,
        if_exists="replace",   # drop + recreate if table already exists
        index=False,
        method="multi",        # batched inserts — faster than row-by-row
        chunksize=10_000,
    )
    print(f"{len(df):,} rows loaded into features table")
    print("\nDatabase initialised successfully.")


if __name__ == "__main__":
    init_db()
