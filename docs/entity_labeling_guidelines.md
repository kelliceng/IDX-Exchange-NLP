# Entity Labeling Guidelines (Week 3)

250 cleaned remarks were labeled by hand: 100 for **dev** (used to write the extraction rules) and 150 for **test** (held out, only used for the final score). Labels are in `data/processed/entity_labels.csv`. Spans point into the `remarks_clean` text.

## What gets labeled

| Entity | Label it when | Example span |
|---|---|---|
| `bedrooms` | The text states how many bedrooms the home has | "4-bedroom", "3 spacious bedrooms" |
| `bathrooms` | The text states how many bathrooms the home has | "2.5 baths", "2 full bathrooms" |
| `sqft` | The text states the home's living area | "1800 square feet" |
| `price` | The text states the current asking price | "offered at $724900" -> "$724900" |

## Rules
- **Only explicit counts.** "Primary suite plus 2 additional bedrooms" is not labeled, since the total (3) is never written.
- **Skip subsets.** Counts described as additional, secondary, guest, or "on each floor" are not the home's total.
- **Main home only.** Counts for an ADU, guest house, or second unit are skipped. If the text gives a total for the whole property first, that total is labeled.
- **First statement wins.** If a remark gives two different counts ("4-bedroom floor plan currently configured as a 3-bedroom"), the first one is labeled.
- **Half baths count as 0.5.** "2 full bathrooms and 2 half bathrooms" = 3.
- **Not square feet.** Lot size, garage, shop, patio, ADU, and planned additions are skipped.
- **Not price.** Upgrade costs, rent, HOA fees, appraisals, nearby sales, and price reductions are skipped.
- **Typos are labeled by meaning.** "2,393 soft of living space" is labeled as 2393 square feet.
