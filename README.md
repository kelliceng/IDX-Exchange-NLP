# IDX Exchange NLP: Real Estate Listing Intelligence

NLP tools for California MLS listing data. Week 1 builds the real estate taxonomy and sample datasets, Week 2 cleans the text, and Week 3 extracts structured facts from it.

## Setup

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and give it at least 4 GB of RAM.
2. Put the SQL dumps (`rets_property.sql` and others) in `data/raw/`. They are not in this repo.
3. Start MySQL. The first start imports the dumps, which takes several minutes.
   ```bash
   docker-compose up -d
   docker-compose logs -f mysql   # wait for "ready for connections"
   ```
4. Install Python packages.
   ```bash
   pip install -r requirements.txt
   ```

## Week 1 workflow

```bash
python scripts/data_loading.py       # 1,000 listing remarks -> data/processed/listing_sample.csv
python scripts/taxonomy_builder.py   # n-gram candidates + data/processed/taxonomy.json
pytest                               # validation tests
```

Then open `notebooks/01_data_exploration.ipynb` for the analysis.

## Week 2 and 3 workflow

```bash
python scripts/text_cleaning.py              # cleaned remarks -> data/processed/listing_sample_clean.csv
python scripts/evaluate_entities.py dev      # entity extraction scores on the dev set
python scripts/evaluate_entities.py test     # final scores on the held-out test set
pytest                                       # all tests (Weeks 1-3)
```

Notebooks: `02_text_cleaning.ipynb` (profiling and before/after examples) and `03_entity_extraction.ipynb` (scores and error analysis).

## Deliverables

| File | What it is |
|---|---|
| `data/processed/taxonomy.json` | 260 real estate terms across 8 categories, each with an ID, synonyms, and sample frequency |
| `data/processed/listing_sample.csv` | 1,000 listing remarks (local only, not committed) |
| `data/processed/sample_queries.csv` | 62 user queries labeled with 7 intents |
| `notebooks/01_data_exploration.ipynb` | Remark patterns, common phrases, and taxonomy coverage |
| `tests/test_week1.py` | Checks taxonomy size, data quality, query labels, and 30%+ coverage |
| `scripts/text_cleaning.py` | Week 2 `TextCleaner` with 9 cleaning steps, a 49-entry abbreviation dictionary, and column profiling |
| `notebooks/02_text_cleaning.ipynb` | Profiling report and before/after examples |
| `tests/test_week2.py` | 70+ cleaning test cases |
| `scripts/entity_extractor.py` | Week 3 `EntityExtractor` for bedrooms, bathrooms, square feet, price, and amenities |
| `data/processed/entity_labels.csv` | 250 hand-labeled remarks (100 dev, 150 test) with entity spans |
| `docs/entity_labeling_guidelines.md` | Rules used for labeling |
| `scripts/evaluate_entities.py` | Precision, recall, and F1 per entity |
| `notebooks/03_entity_extraction.ipynb` | Scores (97% test F1), error analysis, and amenity check |
| `tests/test_week3.py` | Extractor tests plus the 85% F1 target check |

## Taxonomy categories

| Category | Examples |
|---|---|
| `property_type` | condo, townhouse, duplex, manufactured home |
| `architectural_style` | craftsman, mid-century modern, spanish style |
| `rooms_layout` | primary suite, bonus room, ADU, open floor plan |
| `interior_systems` | quartz countertops, vaulted ceilings, solar panels |
| `exterior_lot` | pool, 2-car garage, RV parking, corner lot |
| `community_amenities` | HOA, gated community, clubhouse, Mello-Roos |
| `location_views` | ocean view, walking distance, top-rated schools |
| `condition_sale` | move-in ready, fixer-upper, short sale, rental income |

## Query intents

`search_listings`, `property_details`, `pricing_valuation`, `compare_listings`, `neighborhood_info`, `schedule_showing`, `financing_costs`

## Data policy

MLS data from the company server stays on your machine. `sql-files/`, `data/raw/`, and generated CSVs are in `.gitignore`.
