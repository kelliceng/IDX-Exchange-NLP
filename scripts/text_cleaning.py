"""Clean and standardize listing remarks and user queries.

Run from the project root to clean the Week 1 sample:
    python scripts/text_cleaning.py
"""
import html
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "listing_sample.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "listing_sample_clean.csv"

# Abbreviation -> full form. Matched as whole words, any case.
ABBREVIATIONS = {
    # rooms
    "br": "bedroom", "bd": "bedroom", "bdr": "bedroom", "bdrm": "bedroom", "bdrms": "bedrooms",
    "ba": "bathroom", "bth": "bathroom", "bths": "bathrooms",
    "mbr": "master bedroom", "mstr": "master",
    "rm": "room", "rms": "rooms", "lr": "living room", "fam": "family",
    "bkfst": "breakfast", "bsmt": "basement", "gar": "garage",
    "fp": "fireplace", "frplc": "fireplace",
    # features
    "w/d": "washer/dryer", "a/c": "air conditioning", "ss": "stainless steel",
    "appl": "appliances", "appls": "appliances", "lvp": "luxury vinyl plank",
    "pkg": "parking", "prkg": "parking", "flr": "floor", "flrs": "floors",
    "dbl": "double", "upgr": "upgraded", "sgl": "single", "stry": "story",
    # location
    "fwy": "freeway", "fwys": "freeways", "hwy": "highway", "blvd": "boulevard",
    "ave": "avenue", "blk": "block", "blks": "blocks", "nbhd": "neighborhood",
    # general
    "w/o": "without", "w/": "with", "approx": "approximately", "appx": "approximately",
    "incl": "including", "yr": "year", "yrs": "years", "mins": "minutes",
}

# Acronyms people expect to stay uppercase
KEEP_UPPER = {"HOA", "ADU", "JADU", "HVAC", "LVP", "RV", "EV", "FHA", "USA", "BBQ", "LED", "HDTV"}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


class TextCleaner:
    def __init__(self, abbrev_map=None):
        self.abbrev_map = abbrev_map or ABBREVIATIONS
        # Longest keys first so "w/o" is tried before "w/"
        keys = sorted(self.abbrev_map, key=len, reverse=True)
        alternatives = "|".join(re.escape(k) for k in keys)
        self.abbrev_pattern = re.compile(
            rf"(?<!\w)({alternatives})(?!\w)(?:\.(?=\s+(?-i:[a-z0-9&])))?", re.I
        )

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_case(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_number_words(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        return self.normalize_whitespace(text)

    # ---------- the 9 cleaning steps ----------

    def normalize_unicode(self, text):
        """Fix fancy characters: curly quotes, dashes, hidden spaces, symbols."""
        text = unicodedata.normalize("NFKC", text)  # e.g. ¾ -> 3⁄4, non-breaking space -> space
        replacements = {
            "‘": "'", "’": "'", "“": '"', "”": '"',
            "–": "-", "—": " - ", "⁄": "/",
            "•": " - ", "●": " - ", "·": " - ",
            "…": "...", "±": "", "®": "", "™": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Zero-width characters are invisible but break word matching
        return re.sub(r"[​-‏⁠﻿]", "", text)

    def remove_html(self, text):
        """Remove HTML tags and decode entities like &amp;."""
        text = html.unescape(text)
        text = re.sub(r"<br\s*/?>|</p>", " ", text, flags=re.I)
        return re.sub(r"<[^>]+>", "", text)

    def normalize_whitespace(self, text):
        """Turn line breaks, tabs, and repeated spaces into single spaces."""
        return re.sub(r"\s+", " ", text).strip()

    def normalize_case(self, text):
        """Lowercase SHOUTING: ALL CAPS phrases and ALL CAPS words (4+ letters).
        Known acronyms like HOA and ADU stay uppercase."""
        def fix(match):
            return " ".join(w if w.strip(",.!?") in KEEP_UPPER else w.lower()
                            for w in match.group(0).split(" "))
        # Runs of 2+ ALL CAPS words, e.g. "PRICED AT ONLY", "MUST SEE"
        text = re.sub(r"\b[A-Z][A-Z'&-]*(?:[,.!]? [A-Z][A-Z'&-]*\b){1,}", fix, text)
        # Single ALL CAPS words with 4+ letters, e.g. "CORNER"
        return re.sub(r"\b[A-Z]{4,}\b", fix, text)

    def normalize_punctuation(self, text):
        """Collapse repeated punctuation and remove decoration like ** or ~~."""
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"[*~#]{2,}", "", text)
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # no space before punctuation
        return text

    def normalize_number_words(self, text):
        """'three spacious bedrooms' -> '3 spacious bedrooms', 'two-car garage' -> '2-car garage',
        'four and a half baths' -> '4.5 baths', '2 1/2 baths' -> '2.5 baths'."""
        words = "|".join(NUMBER_WORDS)
        to_digit = lambda w: w if w.isdigit() else str(NUMBER_WORDS[w.lower()])
        # "four and a half" / "1-and-a-half" -> "4.5" / "1.5"
        text = re.sub(
            rf"\b({words}|\d+)[\s-]+and[\s-]+a[\s-]+half\b",
            lambda m: to_digit(m.group(1)) + ".5", text, flags=re.I,
        )
        # Number word followed by up to 2 describing words, then a counted thing.
        # Small words like "of" are not allowed in between, so "one of the bedrooms" is left alone.
        skip = r"(?:of|the|a|an|and|or|out|in|to|with|for|other|more|our|your|its)\b"
        between = rf"(?:[\s-]+(?!{skip})[a-z]+){{0,2}}"
        counted = r"[\s-]+(?:bed|bath|br\b|ba\b|car\b|story|stories|level|unit)"
        text = re.sub(
            rf"\b({words})(?={between}{counted})",
            lambda m: to_digit(m.group(1)), text, flags=re.I,
        )
        # Fractions: "2 1/2 baths" -> "2.5 baths", "1 3/4" -> "1.75"
        fractions = {"1/2": ".5", "1/4": ".25", "3/4": ".75"}
        return re.sub(r"(\d+)[\s-]+(1/2|1/4|3/4)\b", lambda m: m.group(1) + fractions[m.group(2)], text)

    def normalize_prices(self, text):
        """'$1,149,900' -> '$1149900', '450k' -> '450000', '$1.2M' -> '$1200000'."""
        # Remove commas inside dollar amounts
        text = re.sub(r"\$\s?\d{1,3}(?:,\d{3})+", lambda m: m.group(0).replace(",", "").replace(" ", ""), text)
        # "$12 million", "$1.5 mil"
        text = re.sub(
            r"\$(\d+(?:\.\d+)?)\s*(?:million|mil)\b",
            lambda m: "$" + str(round(float(m.group(1)) * 1_000_000)), text, flags=re.I,
        )
        # "$1.2M" (needs the $, so "Residence 4M" is left alone)
        text = re.sub(
            r"\$(\d+(?:\.\d+)?)\s?m{1,2}\b",
            lambda m: "$" + str(round(float(m.group(1)) * 1_000_000)), text, flags=re.I,
        )
        # "450k", "$300K"
        text = re.sub(
            r"(\$?)(\d+(?:\.\d+)?)\s?k\b",
            lambda m: m.group(1) + str(round(float(m.group(2)) * 1_000)), text, flags=re.I,
        )
        return text

    def normalize_measurements(self, text):
        """'2,000 sq. ft.' / '2000sf' / '2,000 SQFT' -> '2000 square feet'."""
        # A trailing "." is only eaten mid-sentence ("sq. ft. of"), not at the end ("sq ft. The")
        mid = r"(?:\.(?=\s+(?-i:[a-z])))?"
        units = rf"(?:sq\.?\s?ft{mid}|sqft|sq\.?\s?feet|square[\s-]?f(?:ee|oo)t|sf\b|s\.f{mid})"
        text = re.sub(
            rf"(\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*-?\s*(?:\+/?-?\s*)?(?:(?:interior|exterior|living|total|finished)\s+)?{units}",
            lambda m: m.group(1).replace(",", "") + " square feet", text, flags=re.I,
        )
        # "0.5 ac" -> "0.5 acres", "12 ft ceilings" -> "12 feet ceilings"
        text = re.sub(r"(\d)\s*ac\b\.?", r"\1 acres", text, flags=re.I)
        text = re.sub(r"(\d)\s*(?:ft\b(?:\.(?=\s+(?-i:[a-z])))?|')(?=[\s.,])", r"\1 feet", text, flags=re.I)
        # "$274 per sq ft" -> "$274 per square foot"
        text = re.sub(rf"\bper\s+{units}", "per square foot", text, flags=re.I)
        return text

    def expand_abbreviations(self, text):
        """'3BR/2BA w/ pool' -> '3 bedroom/2 bathroom with pool'."""
        # Split numbers stuck to abbreviations: "3BR" -> "3 BR"
        text = re.sub(r"(\d)(br|bd|bdrm|ba|bth)\b", r"\1 \2", text, flags=re.I)
        text = self.abbrev_pattern.sub(lambda m: self.abbrev_map[m.group(1).lower()], text)
        # "AC" on its own means air conditioning (number + "ac" was handled as acres)
        text = re.sub(r"\bAC\b", "air conditioning", text)
        # "w/" glued to the next word: "w/pool" -> "with pool"
        return re.sub(r"\bw/(?=[a-z])", "with ", text, flags=re.I)

    # ---------- profiling ----------

    def profile_column(self, df, column_name):
        """Summarize what's in a text column so we know what needs cleaning."""
        col = df[column_name]
        text = col.dropna()
        return {
            "rows": len(col),
            "null_rate": col.isnull().mean(),
            "avg_length": text.str.len().mean(),
            "common_terms": self._extract_top_ngrams(text),
            "price_mentions": int(text.str.contains(r"\$\s?\d").sum()),
            "has_html": int(text.str.contains(r"<[a-zA-Z/][^>]*>|&[a-z]+;").sum()),
            "has_non_ascii": int(text.str.contains(r"[^\x00-\x7f]").sum()),
            "has_zero_width": int(text.str.contains("[​-‏﻿]").sum()),
            "has_line_breaks": int(text.str.contains(r"[\r\n]").sum()),
            "has_all_caps": int(text.str.contains(r"\b[A-Z]{4,}\b").sum()),
            "sqft_mentions": int(text.str.contains(r"sq\.?\s?ft|sqft|\bsf\b|square f", case=False).sum()),
            "common_abbreviations": self._detect_abbreviations(text),
        }

    def _extract_top_ngrams(self, text, n=2, k=10):
        stop = {"the", "and", "a", "of", "to", "in", "with", "for", "is", "this", "on", "your", "an", "from"}
        counts = Counter()
        for t in text:
            words = [w for w in re.findall(r"[a-z]+", t.lower())]
            counts.update(
                " ".join(words[i:i + n]) for i in range(len(words) - n + 1)
                if words[i] not in stop and words[i + n - 1] not in stop
            )
        return counts.most_common(k)

    def _detect_abbreviations(self, text, k=15):
        counts = Counter()
        for t in text:
            counts.update(m.group(1).lower() for m in self.abbrev_pattern.finditer(t))
            counts.update(m.group(2).lower() for m in re.finditer(r"(\d)(br|bd|ba)\b", t, re.I))
        return counts.most_common(k)


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_PATH)
    cleaner = TextCleaner()
    df["remarks_clean"] = df["remarks"].apply(cleaner.clean_text)
    df.to_csv(CLEAN_PATH, index=False)
    changed = (df["remarks"] != df["remarks_clean"]).sum()
    print(f"Cleaned {len(df)} remarks ({changed} changed) -> {CLEAN_PATH.relative_to(ROOT)}")
