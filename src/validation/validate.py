import json
import pandas as pd
from configs.config import RAW, ERRORS, REPORTS

def validate():
    users = pd.read_csv(RAW / "users.csv")
    items = pd.read_csv(RAW / "items.csv")
    interactions = pd.read_csv(RAW / "interactions.csv")

    # Explicit type coercion makes malformed identifiers and values visible.
    users["userId"] = pd.to_numeric(users["userId"], errors="coerce")
    items["movieId"] = pd.to_numeric(items["movieId"], errors="coerce")
    interactions["userId"] = pd.to_numeric(interactions["userId"], errors="coerce")
    interactions["movieId"] = pd.to_numeric(interactions["movieId"], errors="coerce")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    parsed_ts = pd.to_datetime(interactions["timestamp"], errors="coerce", unit="s")
    interactions["_parsed_timestamp"] = parsed_ts

    checks = {}
    checks["users_missing_or_invalid_id"] = int(users["userId"].isna().sum())
    checks["users_missing_source"] = int(users["user_source"].isna().sum()) if "user_source" in users else len(users)
    checks["users_duplicate_id"] = int(users["userId"].duplicated().sum())
    checks["items_missing_or_invalid_id"] = int(items["movieId"].isna().sum())
    checks["items_missing_genres"] = int(items["genres"].isna().sum()) if "genres" in items else len(items)
    checks["items_duplicate_id"] = int(items["movieId"].duplicated().sum())
    checks["items_missing_title"] = int(items["title"].isna().sum())
    checks["interactions_missing_user"] = int(interactions["userId"].isna().sum())
    checks["interactions_missing_item"] = int(interactions["movieId"].isna().sum())
    checks["interactions_missing_rating"] = int(interactions["rating"].isna().sum())
    checks["interactions_missing_timestamp"] = int(interactions["_parsed_timestamp"].isna().sum())
    checks["invalid_rating_range"] = int((~interactions["rating"].between(0.5, 5.0)).fillna(True).sum())
    checks["invalid_timestamp"] = int(interactions["_parsed_timestamp"].isna().sum())
    checks["duplicate_interactions"] = int(interactions.duplicated(["userId","movieId","timestamp"]).sum())

    user_ids = set(users["userId"].dropna())
    item_ids = set(items["movieId"].dropna())
    bad_fk = (~interactions["userId"].isin(user_ids)) | (~interactions["movieId"].isin(item_ids))
    checks["referential_integrity_failures"] = int(bad_fk.sum())

    invalid = (
        interactions["userId"].isna() | interactions["movieId"].isna() |
        interactions["rating"].isna() | ~interactions["rating"].between(0.5,5.0) |
        interactions["_parsed_timestamp"].isna() | bad_fk
    )
    errors = interactions.loc[invalid].copy()
    errors["error_reason"] = ""
    for idx in errors.index:
        reasons = []
        row = interactions.loc[idx]
        if pd.isna(row["userId"]): reasons.append("invalid_or_missing_userId")
        elif row["userId"] not in user_ids: reasons.append("unknown_userId")
        if pd.isna(row["movieId"]): reasons.append("invalid_or_missing_movieId")
        elif row["movieId"] not in item_ids: reasons.append("unknown_movieId")
        if pd.isna(row["rating"]): reasons.append("invalid_or_missing_rating")
        elif not 0.5 <= row["rating"] <= 5.0: reasons.append("rating_out_of_range")
        if pd.isna(row["_parsed_timestamp"]): reasons.append("invalid_timestamp")
        errors.at[idx, "error_reason"] = ";".join(reasons)

    errors.to_csv(ERRORS / "interactions_errors.csv", index=False)
    # Duplicate records are reported and removed in preparation.
    report = {
        "checks": checks,
        "status": "PASS_WITH_REPORTED_ISSUES" if any(checks.values()) else "PASS",
        "input_rows": int(len(interactions)),
        "invalid_interaction_rows": int(len(errors)),
        "note": "Duplicate interaction rows are reported and deduplicated during preparation."
    }
    (REPORTS / "validation_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{"check": k, "failure_count": v} for k,v in checks.items()]).to_csv(
        REPORTS / "validation_checks.csv", index=False
    )
    print("VALIDATION_REPORT:", json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    validate()
