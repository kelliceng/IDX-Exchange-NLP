# IDX Exchange NLP

NLP project for California MLS listing data. The goal is to understand listing descriptions and turn plain-English home searches into database queries.

## Project structure

```
data/
  raw/          SQL dumps (not in repo)
  processed/    taxonomy, sample queries, and labels
  schema.json   database columns, valid cities, and property types
docs/           labeling guidelines
notebooks/      analysis for each week
scripts/        project code
tests/          tests for each week
```

## Setup

1. Install Docker Desktop and give it at least 4 GB of memory.
2. Put the SQL dumps in `data/raw/`.
3. Start MySQL. The first run imports the data and takes a while.
   ```bash
   docker-compose up -d
   ```
4. Install the Python packages.
   ```bash
   pip install -r requirements.txt
   ```
5. Pull the listing sample.
   ```bash
   python scripts/data_loading.py
   ```

## Weekly work

### Week 1: Taxonomy
Real estate terms grouped into 8 categories, plus sample user queries.
- `scripts/taxonomy_builder.py`
- `notebooks/01_data_exploration.ipynb`

### Week 2: Text cleaning
Standardizes listing text (abbreviations, prices, square footage, special characters).
- `scripts/text_cleaning.py`
- `notebooks/02_text_cleaning.ipynb`

### Week 3: Entity extraction
Pulls bedrooms, bathrooms, square footage, price, and amenities from listing descriptions.
- `scripts/entity_extractor.py`
- `scripts/evaluate_entities.py`
- `notebooks/03_entity_extraction.ipynb`

### Week 4: Query parser
Turns searches like "3 bed under 700k in Irvine" into filters and a safe SQL query.
- `scripts/query_parser.py`
- `scripts/run_query.py`
- `notebooks/04_query_parser.ipynb`

## Running tests

```bash
pytest
```

## Data

MLS data stays local and is not committed to this repo.
