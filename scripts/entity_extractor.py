"""Extract bedrooms, bathrooms, square footage, price, and amenities from listing text.

Works best on text cleaned by scripts/text_cleaning.py (e.g. "three bedrooms" -> "3 bedrooms").

    from scripts.entity_extractor import EntityExtractor
    EntityExtractor().extract_all("Charming 3 bedroom, 2 bath home with a pool, 1500 square feet")
"""
import re

from scripts.taxonomy_builder import compile_matchers, load_taxonomy, match_terms

# Taxonomy categories that count as amenities
AMENITY_CATEGORIES = {"interior_systems", "exterior_lot", "community_amenities"}

# Words that mean "not the whole home", e.g. "2 additional bedrooms"
SUBSET_WORDS = r"additional|secondary|guest|other|extra|remaining|more|another|junior"

# Nearby words that mean a count belongs to a second unit, not the main home
OTHER_UNIT = r"adu|guest house|guest home|casita|in-law|mobile home|studio|cottage|apartment|unit with"

# Words after a square footage that mean it's not the living area
NOT_LIVING_AREA = (
    r"lot|parcel|land|garage|shop|workshop|patio|deck|terrace|yard|backyard|"
    r"basement|adu|office|studio|barn|storage|expansion|addition|additional|outdoor"
)

NUM = r"\d+(?:\.\d+)?"
ADJECTIVES = r"(?:[a-z][a-z-]*\s+){0,2}?"  # up to 2 describing words: "3 spacious en-suite bedrooms"


class EntityExtractor:
    def __init__(self, taxonomy=None):
        taxonomy = taxonomy or load_taxonomy()
        amenity_terms = [t for t in taxonomy["terms"] if t["category"] in AMENITY_CATEGORIES]
        self.amenity_matchers = compile_matchers({"terms": amenity_terms})

    # ---------- helpers ----------

    def _first_valid(self, pattern, text, is_valid):
        """Return the first regex match that passes the is_valid check."""
        for match in re.finditer(pattern, text, re.I):
            if is_valid(match, text):
                return match
        return None

    def _is_main_home_count(self, match, text):
        """Reject counts like '2 additional bedrooms' or 'ADU with 1 bedroom'."""
        between = match.group("between") or ""
        before = text[max(0, match.start() - 30): match.start()]
        after = text[match.end(): match.end() + 25]
        checks = [
            # "2 additional bedrooms", "5-piece bathroom"
            re.search(rf"\b(?:{SUBSET_WORDS}|piece)\b", between, re.I),
            # "an additional 2 bedrooms"
            re.search(rf"\b(?:{SUBSET_WORDS})\s*$", before, re.I),
            # "ADU with 1 bedroom", "1 bedroom guest house"
            re.search(rf"\b(?:{OTHER_UNIT})\b", before + " " + after, re.I),
            # "units, each offering 2 bedrooms"
            re.search(r"\beach\b", before, re.I),
            # "can be converted into a 5th bedroom"
            re.search(r"\b(?:converted|potential|optional|could be|can be)\b", before, re.I),
            # "the upper level includes 2 bedrooms" (one floor, not the whole home)
            re.search(r"\b(?:upper|lower|main|first|second|ground|top)\s+(?:level|floor)\b", before, re.I),
            # "3 additional bedrooms, 2 full baths" (baths that go with the extra bedrooms)
            re.search(rf"(?:{SUBSET_WORDS})\s+bed(?:room)?s?\W+$", before, re.I),
        ]
        return not any(checks)

    # ---------- numeric entities ----------

    def extract_bedrooms(self, text):
        pattern = rf"\b(?P<n>\d+)(?!st|nd|rd|th)[\s-]*(?P<between>{ADJECTIVES})(?:bed(?:room)?s?|br|bd)\b"
        match = self._first_valid(pattern, text, self._is_main_home_count)
        return int(match.group("n")) if match else None

    def extract_bathrooms(self, text):
        # "2 full bathrooms and 1 half bath" -> 2.5
        full_half = re.search(
            r"\b(\d+) full (?:bath(?:room)?s?\s*)?(?:and|,|&|\+|plus)\s*(\d+) half", text, re.I
        )
        pattern = rf"\b(?P<n>{NUM})(?!st|nd|rd|th)[\s-]*(?P<between>{ADJECTIVES})(?:bath(?:room)?s?|ba)\b"
        match = self._first_valid(pattern, text, self._is_main_home_count)
        if full_half and (match is None or full_half.start() <= match.start()):
            return int(full_half.group(1)) + 0.5 * int(full_half.group(2))
        if match is None:
            return None
        value = float(match.group("n"))
        return int(value) if value.is_integer() else value

    def extract_sqft(self, text):
        def is_living_area(match, text):
            value = int(match.group("n"))
            after = text[match.end(): match.end() + 25]
            before = text[max(0, match.start() - 40): match.start()]
            if not 300 <= value <= 20000:
                return False
            if re.match(rf"\W*(?:\w+\W+)?(?:{NOT_LIVING_AREA})\b", after, re.I):
                return False
            # "plans to add 1877 square feet", "lot of 7000 square feet"
            if re.search(r"\b(?:add|adding|expand|plans to|lot of|lot size)\b", before, re.I):
                return False
            return True

        match = self._first_valid(r"\b(?P<n>\d{3,5}) square feet", text, is_living_area)
        return int(match.group("n")) if match else None

    def extract_price(self, text):
        # Needs a price word right before the amount: "offered at $724900", "price - only $460000"
        cue = r"(?:price[ds]?|offered|listed|asking|list|now|sale)"
        filler = r"(?:\W+(?:is|at|of|for|to|only|just|reduced|improvement|new))*\W*"
        match = re.search(rf"\b{cue}{filler}\$(\d{{5,}})\b", text, re.I)
        return int(match.group(1)) if match else None

    # ---------- amenities ----------

    def extract_amenities(self, text):
        return [term["term"] for term in match_terms(text, self.amenity_matchers)]

    # ---------- everything ----------

    def extract_all(self, text):
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
            "amenities": self.extract_amenities(text),
        }
