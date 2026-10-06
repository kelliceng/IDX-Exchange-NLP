"""Score the entity extractor against the hand-labeled remarks.

Run from the project root:
    python scripts/evaluate_entities.py          # dev set (used while writing rules)
    python scripts/evaluate_entities.py test     # held-out test set (final score)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.entity_extractor import EntityExtractor

DATA = ROOT / "data" / "processed"
ENTITIES = ["bedrooms", "bathrooms", "sqft", "price"]


def load_labeled(split):
    """One row per labeled remark: cleaned text plus the gold value for each entity."""
    remarks = pd.read_csv(DATA / "entity_label_remarks.csv")
    remarks = remarks[remarks["split"] == split]
    text = pd.read_csv(DATA / "listing_sample_clean.csv").set_index("L_ListingID")["remarks_clean"]
    labels = pd.read_csv(DATA / "entity_labels.csv")

    rows = []
    for listing_id in remarks["listing_id"]:
        gold = labels[labels["listing_id"] == listing_id].set_index("label")["value"].to_dict()
        rows.append({"listing_id": listing_id, "text": text[listing_id],
                     **{e: gold.get(e) for e in ENTITIES}})
    return pd.DataFrame(rows)


def score(df, extractor):
    """Precision, recall, and F1 per entity. A prediction counts only if the value is exactly right."""
    results, errors = {}, []
    for entity in ENTITIES:
        tp = fp = fn = 0
        for _, row in df.iterrows():
            gold = row[entity]
            pred = getattr(extractor, f"extract_{entity}")(row["text"])
            has_gold, has_pred = pd.notna(gold), pred is not None
            if has_gold and has_pred and float(pred) == float(gold):
                tp += 1
            else:
                if has_pred:
                    fp += 1
                if has_gold:
                    fn += 1
                if has_gold or has_pred:
                    errors.append({"listing_id": row["listing_id"], "entity": entity,
                                   "gold": gold, "predicted": pred})
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        results[entity] = {"precision": precision, "recall": recall, "f1": f1,
                           "tp": tp, "fp": fp, "fn": fn, "labeled": int(df[entity].notna().sum())}

    table = pd.DataFrame(results).T
    total = table[["tp", "fp", "fn"]].sum()
    p = total.tp / (total.tp + total.fp)
    r = total.tp / (total.tp + total.fn)
    table.loc["overall"] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r),
                            "tp": total.tp, "fp": total.fp, "fn": total.fn,
                            "labeled": table["labeled"].sum()}
    return table, pd.DataFrame(errors)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    table, errors = score(load_labeled(split), EntityExtractor())
    print(f"\n{split} set results\n")
    print(table.round(3).to_string())
    print(f"\n{len(errors)} errors")
