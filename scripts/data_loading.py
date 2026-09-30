"""Pull a sample of listing remarks from MySQL into a CSV.

Run from the project root once the MySQL container has finished importing:
    python scripts/data_loading.py
"""
from pathlib import Path

import mysql.connector
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "processed" / "listing_sample.csv"

# RAND(42) uses a fixed seed so the same 1000 listings come back every run
QUERY = """
SELECT L_ListingID, L_Address, L_City,
       L_Keyword2    AS beds,
       LM_Dec_3      AS baths,
       L_SystemPrice AS price,
       L_Remarks     AS remarks
FROM rets_property
WHERE L_Remarks IS NOT NULL AND CHAR_LENGTH(L_Remarks) > 50
ORDER BY RAND(42)
LIMIT 1000
"""


def load_sample():
    conn = mysql.connector.connect(
        host="localhost", user="root", password="root", database="real_estate"
    )
    try:
        cursor = conn.cursor()
        cursor.execute(QUERY)
        df = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    df = load_sample()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} listings to {OUTPUT_PATH.relative_to(ROOT)}")
