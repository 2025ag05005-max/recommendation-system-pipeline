from pathlib import Path
import json
import pandas as pd
from configs.config import ROOT

required = [
    "data/raw/users.csv", "data/raw/items.csv", "data/raw/interactions.csv",
    "data/errors/interactions_errors.csv",
    "data/processed/train_interactions.csv", "data/processed/test_interactions.csv",
    "data/features/user_activity_summary.csv", "data/features/item_popularity.csv",
    "data/features/user_item_matrix.csv", "data/features/user_features.csv",
    "data/features/item_features.csv", "data/features/interaction_features.csv",
    "data/features/feature_catalog.csv", "models/popularity_model.joblib",
    "models/item_cf_model.joblib", "reports/validation_report.json",
    "reports/model_comparison.csv", "reports/evaluation_results.json",
    "logs/pipeline_run.json"
]
missing = [p for p in required if not (ROOT/p).exists()]
print("MISSING OUTPUTS:", missing)
if not missing:
    run = json.loads((ROOT/"logs/pipeline_run.json").read_text())
    print("PIPELINE STATUS:", run["status"])
    print("TASKS:", run["tasks"])
    print("MODEL COMPARISON:")
    print(pd.read_csv(ROOT/"reports/model_comparison.csv").to_string(index=False))
    if run["status"] != "SUCCESS":
        raise SystemExit("Pipeline run is not successful.")
else:
    raise SystemExit("Some expected outputs are missing.")
