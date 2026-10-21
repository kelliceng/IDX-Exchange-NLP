"""Turn a plain-English home search into filters and a safe SQL query.

    parser = QueryParser()
    filters = parser.parse("3 bed under 700k in Irvine")
    # {'bedrooms': 3, 'price_max': 700000, 'city': ['Irvine']}

    ok, errors = SchemaValidator().validate_query(filters)
    sql, params = parser.to_sql(filters)
    cursor.execute(sql, params)   # values are passed separately, never pasted into the SQL

Patterns handled (20+):
  price      under / over / between / X to Y / X-Y / around / bare "$850k" / 450k / 1.2M / $1.2 million
  bedrooms   exact / 3+ / at least / or more / up to / 2-3 / two bedrooms
  bathrooms  same as bedrooms, with decimals (2.5)
  sqft       over / under / between / bare "2000 square feet" (treated as a minimum)
  year built after / since / before / newer than
  garage     "2-car garage"
  HOA        "HOA under $300"
  city       known cities, nicknames (LA, SF), several cities, unknown "in <City>" for validation
  type       house, condo, townhome, duplex, ... ; "this condo" is skipped (a specific listing)
  amenities  any Week 1 taxonomy feature; "X or Y" means either one
  negation   "no HOA", "without a pool", "not a condo"
"""
import difflib
import json
import re
from pathlib import Path

from scripts.taxonomy_builder import compile_matchers, load_taxonomy, normalize
from scripts.text_cleaning import TextCleaner

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "data" / "schema.json"

RESULT_COLUMNS = "L_ListingID, L_Address, L_City, L_Type_, L_SystemPrice, L_Keyword2, LM_Dec_3, LM_Int2_3"

CITY_NICKNAMES = {"LA": "Los Angeles", "L.A.": "Los Angeles", "SF": "San Francisco", "SD": "San Diego"}
# City names that are also everyday words only count after "in" or "near"
AMBIGUOUS_CITIES = {"Orange", "Paradise", "Vista", "Nice", "Cool", "Commerce", "Bell", "Sage",
                    "Alpine", "Elk", "Pioneer", "Weed", "Volcano", "Needles", "Bonita", "Onyx"}

PROPERTY_TYPES = {
    r"(?<!guest )(?<!pool )(?<!open )(?<!club )(?<!green )houses?|single[\s-]family(?: homes?| residences?)?|sfr":
        ["SingleFamilyResidence"],
    r"condos?|condominiums?": ["Condominium"],
    r"town\s?houses?|town\s?homes?": ["Townhouse"],
    r"duplex(?:es)?": ["Duplex"],
    r"triplex(?:es)?": ["Triplex"],
    r"fourplex(?:es)?|quadplex(?:es)?|4-plex": ["Quadruplex"],
    r"manufactured homes?|mobile homes?": ["ManufacturedOnLand", "ManufacturedHome", "MobileHome"],
    r"cabins?": ["Cabin"],
    r"co-?ops?": ["StockCooperative"],
    r"lofts?": ["Loft"],
}
# "this condo", "these two townhomes" point at specific listings, so they're not search filters
DEMONSTRATIVE = r"\b(?:this|that|these|those|my|our)\s+(?:\w+\s+)?$"
NEGATION = r"\b(?:no|without|not|non|excluding|except|w/o)[\s-]+(?:an?\s+|any\s+)?$"

# Taxonomy terms that aren't search features
NOT_AMENITIES = {"bedroom", "bathroom", "square feet", "kitchen", "storage", "living room", "den"}

# Amenities with their own yes/no column in rets_property (everything else searches the remarks)
STRUCTURED_AMENITIES = {
    "pool": ("PoolPrivateYN", "1"),
    "spa": ("SpaYN", 1),
    "fireplace": ("FireplaceYN", "1"),
    "garage": ("GarageYN", 1),
    "central air": ("CoolingYN", 1),
    "new construction": ("NewConstructionYN", "1"),
    "55+ community": ("SeniorCommunityYN", 1),
    "single-story": ("StoriesTotal", 1),
    "two-story": ("StoriesTotal", 2),
    "hoa": ("AssociationYN", 1),
}

# Words before a number that make it a maximum, minimum, or rough target
MAX_WORDS = r"under|below|less than|max(?:imum)?|up to|no more than|at most|cheaper than|within"
MIN_WORDS = r"over|above|more than|at least|min(?:imum)?|from|starting at|upwards of"
TARGET_WORDS = r"around|about|approximately|roughly|near|~"
MONEY = r"\$?\s?(\d+(?:\.\d+)?)\s*(k|m|mil|million)?\b"


def _parse_number(number, suffix=""):
    """'450', 'k' -> 450000   '1.2', 'm' -> 1200000"""
    value = float(number)
    suffix = (suffix or "").lower()
    if suffix == "k":
        value *= 1_000
    elif suffix in ("m", "mil", "million"):
        value *= 1_000_000
    return int(value) if value.is_integer() else value


class QueryParser:
    def __init__(self, schema_path=SCHEMA_PATH, taxonomy=None):
        self.cleaner = TextCleaner()
        schema = json.loads(Path(schema_path).read_text())
        self.cities = sorted(schema["cities"], key=len, reverse=True)  # longest first: "West Covina" before "Covina"
        taxonomy = taxonomy or load_taxonomy()
        amenities = [t for t in taxonomy["terms"]
                     if t["category"] != "property_type" and t["term"] not in NOT_AMENITIES]
        self.amenity_terms = compile_matchers({"terms": amenities})
        self.term_variants = {t["term"]: [t["term"]] + t["synonyms"] for t in taxonomy["terms"]}

    # ---------- parse ----------

    def parse(self, query):
        text = self.cleaner.clean_text(query)  # "700k" -> 700000, "three bed" -> 3 bed, "2,000 sqft" -> 2000 square feet
        filters = {}
        text = self._parse_hoa(text, filters)
        text = self._parse_year_built(text, filters)
        self._parse_count(text, filters, "bedrooms", r"bed(?:room)?s?|br")
        self._parse_count(text, filters, "bathrooms", r"bath(?:room)?s?|ba")
        self._parse_sqft(text, filters)
        self._parse_garage(text, filters)
        self._parse_price(text, filters)
        self._parse_property_type(text, filters)
        self._parse_city(query, filters)
        # Blank out city names so "Palm Desert" isn't read as the amenity "desert"
        for city in filters.get("city", []):
            text = re.sub(rf"\b{re.escape(city)}\b", " ", text, flags=re.I)
        text = re.sub(r"\b(?:LA|L\.A\.|SF|SD)\b", " ", text)
        self._parse_amenities(text, filters)
        return filters

    def _parse_hoa(self, text, filters):
        match = re.search(rf"\bhoa(?: fees?| dues)?\s+(?:{MAX_WORDS})\s+{MONEY}", text, re.I)
        if match:
            filters["hoa_max"] = _parse_number(match.group(1), match.group(2))
            text = text[:match.start()] + text[match.end():]  # so it isn't read as the home price
        return text

    def _parse_year_built(self, text, filters):
        for word, key, shift in [("after", "year_built_min", 1), ("since", "year_built_min", 0),
                                 ("newer than", "year_built_min", 1), ("before", "year_built_max", -1),
                                 ("older than", "year_built_max", -1)]:
            match = re.search(rf"\b(?:built\s+)?{word}\s+((?:18|19|20)\d\d)\b", text, re.I)
            if match:
                filters[key] = int(match.group(1)) + shift
                text = text[:match.start()] + text[match.end():]
        return text

    def _parse_count(self, text, filters, name, unit):
        """Bedrooms or bathrooms: exact, minimum, maximum, or range."""
        n = r"(\d+(?:\.\d+)?)"
        unit = rf"[\s-]*(?:{unit})\b"
        checks = [
            (rf"\b(?:between\s+)?{n}\s*(?:-|to|and)\s*{n}{unit}", ("min", "max")),  # 2-3 / between 2 and 3 bedrooms
            (rf"\b(?:at least|minimum(?: of)?|min|over)\s+{n}{unit}", ("min",)),
            (rf"\b{n}\s*\+{unit}", ("min",)),                                # 3+ bed
            (rf"\b{n}{unit}\s*(?:\+|or more|plus)", ("min",)),               # 3 bedrooms or more
            (rf"\b{n}\s+or more{unit}", ("min",)),                           # 3 or more bedrooms
            (rf"\b(?:at most|up to|no more than|max)\s+{n}{unit}", ("max",)),
            (rf"\b{n}{unit}", ("exact",)),                                   # 3 bedroom
        ]
        for pattern, kinds in checks:
            match = re.search(pattern, text, re.I)
            if match:
                for value, kind in zip(match.groups(), kinds):
                    key = name if kind == "exact" else f"{name}_{kind}"
                    filters[key] = _parse_number(value)
                return

    def _parse_sqft(self, text, filters):
        n = r"(\d+)\s*square feet"
        if m := re.search(rf"\bbetween\s+(\d+)\s*(?:square feet\s*)?and\s+{n}", text, re.I):
            filters["sqft_min"], filters["sqft_max"] = int(m.group(1)), int(m.group(2))
        elif m := re.search(rf"\b(?:{MAX_WORDS})\s+{n}", text, re.I):
            filters["sqft_max"] = int(m.group(1))
        elif m := re.search(rf"\b{n}", text, re.I):  # "over 2000 square feet" or just "2000 square feet"
            filters["sqft_min"] = int(m.group(1))

    def _parse_garage(self, text, filters):
        if m := re.search(r"\b(\d+)[\s-]*car garage", text, re.I):
            filters["garage_spaces_min"] = int(m.group(1))

    def _parse_price(self, text, filters):
        # Ignore numbers that belong to square feet, bedrooms, bathrooms, or garages
        text = re.sub(r"\d[\d.]*\s*(?:-|to)?\s*\d*[\s-]*(?:square feet|bed\w*|bath\w*|br\b|ba\b|car\b)", " ", text, flags=re.I)

        def is_price(match):
            number, suffix = match.group(1), match.group(2)
            return "$" in match.group(0) or suffix or float(number) >= 10_000

        def value(match):
            return _parse_number(match.group(1), match.group(2))

        between = rf"\b(?:between\s+)?{MONEY}\s*(?:-|to|and)\s*{MONEY}"
        for m in re.finditer(between, text, re.I):
            first = _parse_number(m.group(1), m.group(2) or m.group(4))   # "500 to 700k" -> both in thousands
            second = _parse_number(m.group(3), m.group(4))
            # "500 to 700k" (the cleaner already turned 700k into 700000): 500 means 500k
            if first < 1_000 <= second:
                first *= 1_000_000 if second >= 1_000_000 and first < 100 else 1_000
            if "$" in m.group(0) or m.group(2) or m.group(4) or second >= 10_000:
                # A reversed range is kept as typed so the validator can point it out
                filters["price_min"], filters["price_max"] = first, second
                return

        for words, key in [(MAX_WORDS, "price_max"), (MIN_WORDS, "price_min"), (TARGET_WORDS, "price_target")]:
            for m in re.finditer(rf"\b(?:{words})\s+{MONEY}", text, re.I):
                if is_price(m):
                    filters[key] = value(m)
                    break

        if not any(k.startswith("price") for k in filters):
            for m in re.finditer(r"\$\s?(\d+(?:\.\d+)?)\s*(k|m|mil|million)?\b", text, re.I):
                filters["price_target"] = value(m)   # "an $850k home"
                break

    def _parse_property_type(self, text, filters):
        include, exclude = [], []
        for pattern, types in PROPERTY_TYPES.items():
            for m in re.finditer(rf"\b(?:{pattern})\b", text, re.I):
                before = text[:m.start()]
                if re.search(DEMONSTRATIVE, before, re.I):
                    continue
                target = exclude if re.search(NEGATION, before, re.I) else include
                target.extend(t for t in types if t not in target)
        if include:
            filters["property_type"] = include
        if exclude:
            filters["exclude_property_type"] = exclude

    def _parse_city(self, query, filters):
        found, taken = [], []

        def add(city, start, end):
            if city not in found and not any(s < end and start < e for s, e in taken):
                found.append(city)
                taken.append((start, end))

        for nickname, city in CITY_NICKNAMES.items():
            for m in re.finditer(rf"(?<![\w.]){re.escape(nickname)}(?![\w])", query):
                add(city, m.start(), m.end())
        for city in self.cities:
            for m in re.finditer(rf"\b{re.escape(city)}\b", query, re.I):
                after_in = re.search(r"\b(?:in|near)\s+$", query[:m.start()], re.I)
                typed_capitalized = query[m.start()].isupper()
                if city in AMBIGUOUS_CITIES and not after_in:
                    continue
                if typed_capitalized or after_in:
                    add(city, m.start(), m.end())
        # "in Portland": a capitalized place we don't know. Kept so the validator can flag it.
        for m in re.finditer(r"\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", query):
            if not any(s <= m.start(1) < e for s, e in taken):
                add(m.group(1), m.start(1), m.end(1))
        # "in irvin": lowercase and misspelled, but close to a real city. Kept so the validator can suggest it.
        lowered = {c.lower() for c in self.cities}
        for m in re.finditer(r"\bin\s+([a-z]+(?:\s+[a-z]+)?)\b", query):
            for words in (m.group(1), m.group(1).split()[0]):
                if difflib.get_close_matches(words, lowered, n=1, cutoff=0.88) and \
                        not any(s <= m.start(1) < e for s, e in taken):
                    add(words, m.start(1), m.start(1) + len(words))
                    break
        if found:
            filters["city"] = found

    def _parse_amenities(self, text, filters):
        lowered = normalize(text)
        spans = []
        for term, regex in self.amenity_terms:
            if m := regex.search(lowered):
                spans.append((m.start(), m.end(), term["term"]))
        # Keep the longest match when terms overlap: "mid-century modern" beats "modern"
        spans.sort(key=lambda s: s[0] - s[1])
        kept = []
        for start, end, term in spans:
            if not any(s < end and start < e for s, e, _ in kept):
                kept.append((start, end, term))
        kept.sort()

        include, any_of, exclude = [], [], []
        for i, (start, end, term) in enumerate(kept):
            if term == "garage" and "garage_spaces_min" in filters:
                continue
            if term == "no hoa":
                exclude.append("hoa")
                continue
            if re.search(NEGATION, lowered[:start]):
                exclude.append(term)
                continue
            # "an ADU or guest house" -> either one is fine
            joined_by_or = lambda a, b: re.fullmatch(r"\s*,?\s*or\s+(?:an?\s+|the\s+)?", lowered[a:b])
            if (i + 1 < len(kept) and joined_by_or(end, kept[i + 1][0])) or \
               (i > 0 and joined_by_or(kept[i - 1][1], start)):
                any_of.append(term)
            else:
                include.append(term)
        if include:
            filters["amenities"] = include
        if any_of:
            filters["amenities_any"] = any_of
        if exclude:
            filters["exclude_amenities"] = exclude

    # ---------- SQL ----------

    def to_sql(self, filters, limit=50):
        """Build a parameterized query. Column names come from a fixed list and
        every user value goes into params, so nothing typed by a user is pasted into the SQL."""
        conditions, params = [], []

        def add(condition, *values):
            conditions.append(condition)
            params.extend(values)

        numeric = {
            "price_min": "L_SystemPrice >= %s", "price_max": "L_SystemPrice <= %s",
            "bedrooms": "L_Keyword2 = %s", "bedrooms_min": "L_Keyword2 >= %s", "bedrooms_max": "L_Keyword2 <= %s",
            "bathrooms": "LM_Dec_3 = %s", "bathrooms_min": "LM_Dec_3 >= %s", "bathrooms_max": "LM_Dec_3 <= %s",
            "sqft_min": "LM_Int2_3 >= %s", "sqft_max": "LM_Int2_3 <= %s",
            "year_built_min": "YearBuilt >= %s", "year_built_max": "YearBuilt <= %s",
            "garage_spaces_min": "L_Keyword5 >= %s", "hoa_max": "AssociationFee <= %s",
        }
        list_filters = {"city", "property_type", "exclude_property_type",
                        "amenities", "amenities_any", "exclude_amenities", "price_target"}
        unknown = set(filters) - set(numeric) - list_filters
        if unknown:
            raise ValueError(f"Unknown filter(s): {sorted(unknown)}")

        for key, condition in numeric.items():
            if key in filters:
                add(condition, filters[key])
        if "price_target" in filters:  # within 10% of the price mentioned
            add("L_SystemPrice BETWEEN %s AND %s",
                round(filters["price_target"] * 0.9), round(filters["price_target"] * 1.1))
        if "city" in filters:
            add(f"L_City IN ({', '.join(['%s'] * len(filters['city']))})", *filters["city"])
        if "property_type" in filters:
            types = filters["property_type"]
            add(f"L_Type_ IN ({', '.join(['%s'] * len(types))})", *types)
        if "exclude_property_type" in filters:
            types = filters["exclude_property_type"]
            add(f"L_Type_ NOT IN ({', '.join(['%s'] * len(types))})", *types)
        for term in filters.get("amenities", []):
            add(*self._amenity_condition(term))
        if filters.get("amenities_any"):
            parts = [self._amenity_condition(t) for t in filters["amenities_any"]]
            add("(" + " OR ".join(c for c, *_ in parts) + ")", *[v for _, *vs in parts for v in vs])
        for term in filters.get("exclude_amenities", []):
            add(*self._amenity_condition(term, exclude=True))

        sql = f"SELECT {RESULT_COLUMNS} FROM rets_property"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " LIMIT %s"
        params.append(int(limit))
        return sql, params

    def _amenity_condition(self, term, exclude=False):
        """Returns (condition, value, value, ...) for one amenity."""
        if term in STRUCTURED_AMENITIES:
            column, value = STRUCTURED_AMENITIES[term]
            if exclude:
                return f"({column} IS NULL OR {column} <> %s)", value
            return f"{column} = %s", value
        # Not a database column, so search the listing description for the term or a synonym
        variants = self.term_variants.get(term, [term])[:6]
        likes = [f"%{v}%" for v in variants]
        if exclude:
            return "(L_Remarks IS NULL OR (" + " AND ".join(["L_Remarks NOT LIKE %s"] * len(likes)) + "))", *likes
        return "(" + " OR ".join(["L_Remarks LIKE %s"] * len(likes)) + ")", *likes


class SchemaValidator:
    """Checks parsed filters against the real data before any SQL is run."""

    def __init__(self, schema_path=SCHEMA_PATH):
        self.schema = json.loads(Path(schema_path).read_text())
        self.valid_cities = {c.lower(): c for c in self.schema["cities"]}
        self.valid_types = set(self.schema["property_types"])

    def validate_query(self, filters):
        errors = []
        ranges = self.schema["filters"]

        for key, value in filters.items():
            if key not in ranges:
                errors.append(f"Unknown filter '{key}'")
                continue
            low, high = ranges[key].get("min"), ranges[key].get("max")
            if low is not None and not (low <= value <= high):
                errors.append(f"{key.replace('_', ' ').capitalize()} {value:,} is outside the expected range "
                              f"({low:,} to {high:,})")

        for city in filters.get("city", []):
            if city.lower() not in self.valid_cities:
                close = difflib.get_close_matches(city.lower(), self.valid_cities, n=1, cutoff=0.88)
                hint = f" Did you mean {self.valid_cities[close[0]]}?" if close else ""
                errors.append(f"City '{city}' not found in the database.{hint}")

        for key in ("property_type", "exclude_property_type"):
            for t in filters.get(key, []):
                if t not in self.valid_types:
                    errors.append(f"Property type '{t}' not found in the database")

        for low_key, high_key in [("price_min", "price_max"), ("bedrooms_min", "bedrooms_max"),
                                  ("bathrooms_min", "bathrooms_max"), ("sqft_min", "sqft_max"),
                                  ("year_built_min", "year_built_max")]:
            if low_key in filters and high_key in filters and filters[low_key] > filters[high_key]:
                errors.append(f"{low_key} is greater than {high_key}")

        return len(errors) == 0, errors
