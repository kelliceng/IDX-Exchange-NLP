"""Pull listing remarks from MySQL into a CSV.

Run from the project root once the MySQL container has finished importing:
    python scripts/data_loading.py         # 1,000 random listings -> listing_sample.csv
    python scripts/data_loading.py --all   # every listing with remarks -> all_listings.csv (Week 5)
"""
import sys
from pathlib import Path

import mysql.connector
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "listing_sample.csv"
ALL_PATH = ROOT / "data" / "processed" / "all_listings.csv"

QUERY = """
SELECT L_ListingID, L_Address, L_City,
       L_Keyword2    AS beds,
       LM_Dec_3      AS baths,
       L_SystemPrice AS price,
       L_Remarks     AS remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL AND CHAR_LENGTH(L_Remarks) > 50
"""

# RAND(42) uses a fixed seed so the same 1000 listings come back every run
SAMPLE_QUERY = QUERY + "ORDER BY RAND(42) LIMIT 1000"


def run_query(query):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="real_estate"
    )
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
    finally:
        conn.close()
    return df


def load_sample():
    return run_query(SAMPLE_QUERY)


def load_all():
    return run_query(QUERY)


if __name__ == "__main__":
    if "--all" in sys.argv:
        df, path = load_all(), ALL_PATH
    else:
        df, path = load_sample(), SAMPLE_PATH
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} listings to {path.relative_to(ROOT)}")
