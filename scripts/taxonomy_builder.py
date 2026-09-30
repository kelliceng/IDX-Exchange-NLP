"""Build the real estate terminology taxonomy.

Step 1  Count the most common n-grams in listing remarks. These are the
        candidate terms, saved to data/processed/ngram_candidates.csv.
Step 2  Write the curated taxonomy (TERMS below) to data/processed/taxonomy.json,
        with how many sample remarks mention each term.

Run from the project root:
    python scripts/taxonomy_builder.py
"""
import json
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.util import ngrams

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "processed" / "listing_sample.csv"
TAXONOMY_PATH = ROOT / "data" / "processed" / "taxonomy.json"
CANDIDATES_PATH = ROOT / "data" / "processed" / "ngram_candidates.csv"

# category name -> (id prefix, description)
CATEGORIES = {
    "property_type": ("PT", "What kind of property it is"),
    "architectural_style": ("AS", "Design style and number of stories"),
    "rooms_layout": ("RL", "Rooms, spaces, and floor plan"),
    "interior_systems": ("IN", "Finishes, fixtures, appliances, and home systems"),
    "exterior_lot": ("EX", "Yard, outdoor living, parking, and lot"),
    "community_amenities": ("CA", "HOA and shared community features"),
    "location_views": ("LV", "Views, surroundings, and nearby places"),
    "condition_sale": ("CS", "Condition, upgrades, and sale terms"),
}

# The first word or phrase is the canonical term. The rest are synonyms and
# abbreviations that mean the same thing. Plurals are matched automatically.
TERMS = {
    "property_type": [
        "single family home|single-family|single family residence|sfr|detached home",
        "condo|condominium",
        "townhouse|townhome|town home",
        "duplex",
        "triplex",
        "fourplex|quadplex|4-plex",
        "multi-family|multifamily|multi-unit|income property",
        "apartment|apartment building",
        "manufactured home|mobile home",
        "cabin",
        "co-op|cooperative|stock cooperative",
        "loft condo|live/work loft",
        "studio apartment|studio unit|studio condo",
        "bungalow",
        "cottage",
        "estate home|estate property|luxury estate",
        "villa",
        "penthouse",
        "mixed use",
        "vacant land|buildable lot|land parcel",
        "patio home",
        "high-rise|highrise",
        "horse property|equestrian property",
        "farm|agricultural property",
        "planned unit development|pud",
        "end unit",
        "tract home",
        "twin home",
    ],
    "architectural_style": [
        "modern",
        "contemporary",
        "mid-century modern|mid-century|mcm",
        "craftsman",
        "spanish style|spanish revival|spanish colonial",
        "mediterranean",
        "tuscan",
        "colonial",
        "victorian",
        "ranch style|ranch-style home",
        "cape cod",
        "farmhouse|modern farmhouse",
        "traditional",
        "tudor",
        "art deco",
        "french country|french provincial",
        "coastal style|coastal-inspired|beach-style",
        "transitional",
        "minimalist",
        "industrial",
        "split-level",
        "tri-level",
        "two-story|2-story",
        "single-story|one-story|single-level|single level",
        "a-frame",
        "hacienda",
        "mission style|mission revival",
    ],
    "rooms_layout": [
        "primary suite|primary bedroom|master suite|master bedroom|owner's suite",
        "bedroom|br|bd",
        "bathroom|bath|ba",
        "half bath|powder room|half bathroom",
        "en-suite|ensuite",
        "walk-in closet",
        "family room",
        "living room",
        "great room",
        "formal dining room|formal dining",
        "dining room|dining area|dining nook|breakfast nook",
        "kitchen",
        "home office|office|study",
        "den",
        "bonus room",
        "loft space|upstairs loft",
        "media room|theater room|home theater",
        "game room|rec room|recreation room",
        "home gym|exercise room",
        "laundry room|laundry area|inside laundry|indoor laundry|in-unit laundry",
        "mudroom",
        "basement",
        "attic",
        "wine cellar|wine room",
        "guest house|casita|pool house",
        "in-law suite|guest suite|junior suite|mother-in-law",
        "adu|accessory dwelling unit|granny flat|jadu",
        "open floor plan|open-concept|open concept|open layout",
        "split floor plan|split bedroom",
        "main-level bedroom|downstairs bedroom|bedroom on the main|first-floor bedroom",
        "flex space|flex room",
        "sunroom",
        "pantry|walk-in pantry|butler's pantry",
        "foyer|entryway",
        "storage",
        "multigenerational|multi-generational|dual living",
        "dual primary suites|two primary suites",
        "square feet|sq ft|sqft|square footage",
    ],
    "interior_systems": [
        "hardwood floors|hardwood flooring|hardwood",
        "luxury vinyl plank|lvp|vinyl plank|luxury vinyl",
        "laminate flooring|laminate",
        "tile flooring|tile floors|porcelain tile|ceramic tile",
        "carpet",
        "vaulted ceilings|cathedral ceilings",
        "high ceilings|soaring ceilings|double-height ceilings|volume ceilings",
        "recessed lighting|recessed lights|can lights",
        "crown molding|crown moulding",
        "fireplace|wood-burning fireplace|gas fireplace",
        "quartz countertops|quartz",
        "granite countertops|granite",
        "marble",
        "butcher block",
        "center island|kitchen island|island",
        "breakfast bar|eat-in kitchen",
        "stainless steel appliances|stainless appliances|stainless steel",
        "gourmet kitchen|chef's kitchen|chef kitchen|chefs kitchen",
        "custom cabinetry|custom cabinets|shaker cabinets|soft-close cabinets",
        "built-ins|built-in shelving|built-in cabinets",
        "dual vanities|double vanity|dual sinks|double sinks",
        "soaking tub|freestanding tub|jetted tub",
        "walk-in shower|frameless shower|rain shower",
        "skylight",
        "natural light|abundant natural light|light-filled",
        "dual-pane windows|double-pane windows|dual pane|energy-efficient windows",
        "plantation shutters",
        "ceiling fan",
        "wet bar",
        "washer and dryer|washer/dryer",
        "central air|central ac|air conditioning|a/c|hvac",
        "forced-air heating|forced air|central heat|central heating",
        "mini-split|ductless",
        "tankless water heater|tankless",
        "solar panels|solar|owned solar|leased solar",
        "ev charger|ev charging|electric vehicle charger|220v",
        "smart home|smart thermostat",
        "water softener|water filtration|reverse osmosis",
        "security system|alarm system|security cameras",
        "elevator",
        "wine fridge|wine refrigerator|beverage fridge",
        "sliding glass doors|sliding doors|french doors",
    ],
    "exterior_lot": [
        "pool|private pool|swimming pool|saltwater pool|heated pool",
        "spa|hot tub|jacuzzi",
        "backyard|back yard|rear yard",
        "front yard",
        "patio|covered patio",
        "deck|rooftop deck|roof deck",
        "balcony|private balcony",
        "outdoor kitchen|built-in bbq|bbq|barbecue",
        "fire pit|firepit",
        "garage|attached garage|detached garage|two-car garage|three-car garage",
        "carport",
        "driveway|circular driveway",
        "rv parking|rv access|boat parking",
        "parking|assigned parking|off-street parking|tandem parking",
        "fenced yard|fully fenced|fenced",
        "landscaping|mature landscaping|drought-tolerant landscaping|professionally landscaped",
        "fruit trees|citrus trees|orchard",
        "garden|raised garden beds|vegetable garden",
        "lawn|artificial turf|synthetic turf",
        "corner lot",
        "cul-de-sac|cul de sac",
        "large lot|oversized lot|expansive lot",
        "acre|acreage",
        "sport court|basketball court",
        "workshop|shed|storage shed",
        "porch|front porch|wrap-around porch",
        "pergola|gazebo",
        "courtyard",
        "curb appeal",
        "putting green",
        "outdoor shower",
        "indoor-outdoor living|indoor/outdoor living|indoor outdoor flow",
    ],
    "community_amenities": [
        "hoa|homeowners association|association fee|hoa dues",
        "no hoa",
        "low hoa",
        "gated community|gated|guard-gated|24-hour security",
        "community pool|association pool",
        "clubhouse|club house",
        "fitness center|fitness room|community gym",
        "tennis courts|tennis court|tennis",
        "pickleball",
        "golf course|golf",
        "playground|tot lot",
        "dog park|pet-friendly|pet friendly",
        "walking trails|hiking trails|biking trails|trails",
        "greenbelt|green belt|open space|community park",
        "picnic area|bbq area",
        "concierge|doorman",
        "guest parking",
        "55+ community|senior community|age-restricted|active adult",
        "master-planned community|master planned community|planned community",
        "resort-style amenities|resort-style|resort style",
        "sauna|steam room",
        "business center",
        "recreation center|rec center|community center",
        "private beach|beach club",
        "marina|boat slip|boat dock",
        "equestrian|horse trails",
        "mello-roos|mello roos",
    ],
    "location_views": [
        "ocean view|ocean-view",
        "mountain view",
        "city lights view|city view|city lights",
        "panoramic view|sweeping view",
        "canyon view",
        "golf course view",
        "lake view",
        "water view",
        "bay view",
        "beachfront|oceanfront|beach front|ocean front",
        "waterfront|lakefront",
        "near the beach|close to the beach|steps to the beach|walk to the beach|beach access",
        "walking distance|walkable",
        "downtown",
        "shopping|shopping center",
        "restaurants|cafes",
        "schools|top-rated schools|award-winning schools|school district|blue ribbon",
        "freeway access|freeways|highway access|easy commute|commuter",
        "public transportation|public transit|metro|light rail|train station",
        "parks|nearby parks",
        "quiet neighborhood|quiet street|peaceful neighborhood|tree-lined street",
        "hillside|hilltop",
        "airport",
        "hospitals|medical center",
        "university|college",
        "rural|country living|rural setting",
        "urban|city living|urban living",
        "desert",
        "wine country|vineyard",
        "ski resort|ski",
        "coastal|coastal living",
    ],
    "condition_sale": [
        "move-in ready|turnkey|turn-key",
        "remodeled|fully remodeled|renovated|fully renovated|renovation",
        "updated|recently updated|upgraded|upgrades",
        "new construction|newly built|brand new home",
        "fixer-upper|fixer|handyman special|needs work|tlc",
        "as-is|sold as-is",
        "investment opportunity|investment property|investor special|investors",
        "rental income|income-producing|cash flow|rental potential|tenant-occupied",
        "vacant",
        "well-maintained|meticulously maintained|pride of ownership|lovingly maintained",
        "original owner|one owner|original condition",
        "new roof|newer roof",
        "new hvac|newer hvac|new ac",
        "fresh paint|freshly painted|new paint|fresh interior paint|new interior paint",
        "new flooring|new floors|new carpet",
        "repiped|copper plumbing|pex plumbing|new plumbing",
        "permitted|unpermitted|non-permitted",
        "short sale",
        "reo|bank-owned|foreclosure",
        "probate|trust sale|estate sale",
        "seller financing|owner financing|owner will carry",
        "assumable loan|assumable",
        "price reduced|price reduction|new price",
        "motivated seller|bring all offers|all offers considered",
        "multiple offers|offer deadline|offers due",
        "cash only|cash offers",
        "fha|va loan",
        "backup offers|contingent|pending",
        "open house",
        "virtual tour|3d tour|matterport",
        "private showing|schedule a showing|by appointment",
        "furnished|fully furnished|unfurnished",
        "custom-built|custom built|custom home",
        "energy efficient|energy-efficient|green certified|leed",
        "buyer to verify|buyer to investigate",
    ],
}


# ---------- matching helpers (used by tests and the notebook too) ----------

def normalize(text):
    """Lowercase and straighten curly apostrophes."""
    return str(text).lower().replace("’", "'").replace("‘", "'")


def variant_pattern(variant):
    """'walk-in closet' also matches 'walk in closet', 'walkin closet', 'walk-in closets'."""
    words = re.split(r"[\s\-]+", normalize(variant))
    body = r"[\s\-]?".join(re.escape(w) for w in words)
    return rf"(?<![a-z0-9]){body}s?(?![a-z0-9])"


def compile_matchers(taxonomy):
    """Return a list of (term, compiled regex) pairs, one per taxonomy term."""
    matchers = []
    for term in taxonomy["terms"]:
        variants = [term["term"]] + term["synonyms"]
        pattern = "|".join(variant_pattern(v) for v in variants)
        matchers.append((term, re.compile(pattern)))
    return matchers


def match_terms(text, matchers):
    """Return the taxonomy terms that appear in a piece of text."""
    text = normalize(text)
    return [term for term, regex in matchers if regex.search(text)]


def load_taxonomy(path=TAXONOMY_PATH):
    with open(path) as f:
        return json.load(f)


# ---------- step 1: n-gram candidates ----------

def top_ngrams(remarks, n, k):
    """Most common n-grams that don't start or end with a stopword."""
    stop = set(stopwords.words("english"))
    tokens = nltk.word_tokenize(normalize(" ".join(remarks.dropna())))
    tokens = [t for t in tokens if re.fullmatch(r"[a-z][a-z\-']*", t)]
    grams = (g for g in ngrams(tokens, n) if g[0] not in stop and g[-1] not in stop)
    return Counter(grams).most_common(k)


def ngram_candidates(remarks, taxonomy, k=200):
    matchers = compile_matchers(taxonomy)
    rows = []
    for n in (1, 2, 3):
        for gram, count in top_ngrams(remarks, n, k):
            phrase = " ".join(gram)
            rows.append({
                "ngram": phrase,
                "n": n,
                "count": count,
                "in_taxonomy": bool(match_terms(phrase, matchers)),
            })
    return pd.DataFrame(rows)


# ---------- step 2: build the taxonomy ----------

def build_taxonomy(remarks=None):
    terms = []
    for category, entries in TERMS.items():
        prefix = CATEGORIES[category][0]
        for i, entry in enumerate(entries, start=1):
            canonical, *synonyms = entry.split("|")
            terms.append({
                "id": f"{prefix}{i:03d}",
                "term": canonical,
                "category": category,
                "synonyms": synonyms,
            })

    taxonomy = {
        "version": "1.0",
        "categories": {name: desc for name, (_, desc) in CATEGORIES.items()},
        "terms": terms,
    }

    # Count how many sample remarks mention each term
    if remarks is not None:
        texts = [normalize(t) for t in remarks.dropna()]
        for term, regex in compile_matchers(taxonomy):
            term["sample_count"] = sum(1 for t in texts if regex.search(t))

    return taxonomy


def main():
    remarks = None
    if SAMPLE_PATH.exists():
        remarks = pd.read_csv(SAMPLE_PATH)["remarks"]
    else:
        print("No listing sample yet. Run scripts/data_loading.py first for term counts.")

    taxonomy = build_taxonomy(remarks)
    with open(TAXONOMY_PATH, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"Saved {len(taxonomy['terms'])} terms in "
          f"{len(taxonomy['categories'])} categories to {TAXONOMY_PATH.relative_to(ROOT)}")

    if remarks is not None:
        candidates = ngram_candidates(remarks, taxonomy)
        candidates.to_csv(CANDIDATES_PATH, index=False)
        missing = candidates[(candidates.n > 1) & ~candidates.in_taxonomy].head(15)
        print(f"\nSaved n-gram candidates to {CANDIDATES_PATH.relative_to(ROOT)}")
        print("Frequent phrases not yet in the taxonomy (review these to expand it):")
        for _, row in missing.iterrows():
            print(f"  {row.ngram}: {row['count']}")


if __name__ == "__main__":
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    main()
