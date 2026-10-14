"""Try a plain-English search against the MySQL database.

Run from the project root while MySQL is running:
    python scripts/run_query.py "3 bed under 700k in Irvine with a pool"
"""
import sys
from pathlib import Path

import mysql.connector
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.query_parser import QueryParser, SchemaValidator

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "3 bed under 700k in Irvine"
    parser, validator = QueryParser(), SchemaValidator()

    filters = parser.parse(query)
    print(f"Query:   {query}\nFilters: {filters}")

    ok, errors = validator.validate_query(filters)
    if not ok:
        print("Can't run this search:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    sql, params = parser.to_sql(filters, limit=10)
    print(f"SQL:     {sql}\nParams:  {params}\n")

    conn = mysql.connector.connect(host="localhost", user="root", password="root", database="real_estate")
    cursor = conn.cursor()
    cursor.execute(sql, params)  # values are sent separately from the SQL text
    results = pd.DataFrame(cursor.fetchall(), columns=cursor.column_names)
    conn.close()
    print(results.to_string(index=False) if len(results) else "No matching listings.")
