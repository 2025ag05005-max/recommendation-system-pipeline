import pandas as pd
from configs.config import RAW, PROCESSED

def prepare():
    users = pd.read_csv(RAW / "users.csv").drop_duplicates("userId")
    items = pd.read_csv(RAW / "items.csv").drop_duplicates("movieId")
    interactions = pd.read_csv(RAW / "interactions.csv")

    interactions["userId"] = pd.to_numeric(interactions["userId"], errors="coerce")
    interactions["movieId"] = pd.to_numeric(interactions["movieId"], errors="coerce")
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions["timestamp"] = pd.to_datetime(
        pd.to_numeric(interactions["timestamp"], errors="coerce"), unit="s", errors="coerce"
    )
    interactions = interactions.dropna(subset=["userId","movieId","rating","timestamp"])
    interactions = interactions[interactions["rating"].between(0.5,5.0)]
    interactions = interactions[
        interactions["userId"].isin(users["userId"]) &
        interactions["movieId"].isin(items["movieId"])
    ]
    interactions = interactions.drop_duplicates(["userId","movieId","timestamp"])
    interactions["userId"] = interactions["userId"].astype(int)
    interactions["movieId"] = interactions["movieId"].astype(int)
    interactions["rating"] = interactions["rating"].astype(float)
    interactions["interaction_date"] = interactions["timestamp"].dt.strftime("%Y-%m-%d")

    # Per-user temporal holdout: latest interaction becomes test; earlier events train.
    interactions = interactions.sort_values(["userId","timestamp","movieId"])
    counts = interactions.groupby("userId")["movieId"].transform("count")
    test_mask = (counts >= 2) & (
        interactions["timestamp"] == interactions.groupby("userId")["timestamp"].transform("max")
    )
    # Hold out every event at the latest timestamp for each eligible user.
    # This avoids placing same-time events in both train and test.
    test_rows = interactions.loc[test_mask]
    test_index = set(test_rows.index)
    test = interactions.loc[interactions.index.isin(test_index)].copy()
    train = interactions.loc[~interactions.index.isin(test_index)].copy()

    users.to_csv(PROCESSED / "users_clean.csv", index=False)
    items.to_csv(PROCESSED / "items_clean.csv", index=False)
    train.to_csv(PROCESSED / "train_interactions.csv", index=False)
    test.to_csv(PROCESSED / "test_interactions.csv", index=False)
    interactions.to_csv(PROCESSED / "interactions_clean.csv", index=False)

    print({
        "clean_interactions": len(interactions), "train_rows": len(train),
        "test_rows": len(test), "users": len(users), "items": len(items),
        "train_max_timestamp": str(train["timestamp"].max()),
        "test_min_timestamp": str(test["timestamp"].min())
    })
    return train, test, users, items

if __name__ == "__main__":
    prepare()
