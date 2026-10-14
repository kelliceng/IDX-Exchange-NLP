"""Build data/schema.json: which database column each search filter uses, valid ranges,
and the lists of real cities and property types (pulled from MySQL).

Run from the project root once MySQL is running:
    python scripts/build_schema.py
"""
import json
from pathlib import Path

import mysql.connector

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "data" / "schema.json"

# filter name -> database column and the range of values that make sense
FILTERS = {
    "price_min":         {"column": "L_SystemPrice", "min": 100_000, "max": 10_000_000},
    "price_max":         {"column": "L_SystemPrice", "min": 100_000, "max": 10_000_000},
    "price_target":      {"column": "L_SystemPrice", "min": 100_000, "max": 10_000_000},
    "bedrooms":          {"column": "L_Keyword2", "min": 0, "max": 10},
    "bedrooms_min":      {"column": "L_Keyword2", "min": 0, "max": 10},
    "bedrooms_max":      {"column": "L_Keyword2", "min": 0, "max": 10},
    "bathrooms":         {"column": "LM_Dec_3", "min": 1, "max": 10},
    "bathrooms_min":     {"column": "LM_Dec_3", "min": 1, "max": 10},
    "bathrooms_max":     {"column": "LM_Dec_3", "min": 1, "max": 10},
    "sqft_min":          {"column": "LM_Int2_3", "min": 200, "max": 20_000},
    "sqft_max":          {"column": "LM_Int2_3", "min": 200, "max": 20_000},
    "year_built_min":    {"column": "YearBuilt", "min": 1850, "max": 2030},
    "year_built_max":    {"column": "YearBuilt", "min": 1850, "max": 2030},
    "garage_spaces_min": {"column": "L_Keyword5", "min": 1, "max": 10},
    "hoa_max":           {"column": "AssociationFee", "min": 0, "max": 5_000},
    "city":              {"column": "L_City"},
    "property_type":     {"column": "L_Type_"},
    "exclude_property_type": {"column": "L_Type_"},
    "amenities":         {"column": "(mapped per amenity)"},
    "amenities_any":     {"column": "(mapped per amenity)"},
    "exclude_amenities": {"column": "(mapped per amenity)"},
}


def distinct(cursor, column):
    cursor.execute(f"SELECT DISTINCT {column} FROM rets_property WHERE {column} IS NOT NULL")
    return sorted(row[0] for row in cursor.fetchall())


if __name__ == "__main__":
    conn = mysql.connector.connect(host="localhost", user="root", password="root", database="real_estate")
    cursor = conn.cursor()
    schema = {
        "table": "rets_property",
        "filters": FILTERS,
        "cities": distinct(cursor, "L_City"),
        "property_types": distinct(cursor, "L_Type_"),
    }
    conn.close()
    SCHEMA_PATH.write_text(json.dumps(schema, indent=1))
    print(f"Saved {len(schema['cities'])} cities and {len(schema['property_types'])} property types to {SCHEMA_PATH.relative_to(ROOT)}")
