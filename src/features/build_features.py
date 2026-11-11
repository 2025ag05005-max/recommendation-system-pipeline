import pandas as pd
from configs.config import PROCESSED, FEATURES


def build_features():
    FEATURES.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(
        PROCESSED / "train_interactions.csv",
        parse_dates=["timestamp"]
    )

    items = pd.read_csv(PROCESSED / "items_clean.csv")

    global_last = train["timestamp"].max()

    # -------------------------------------------------
    # 1. USER FEATURES
    # -------------------------------------------------
    user = (
        train.groupby("userId")
        .agg(
            user_activity_count=("movieId", "count"),
            user_average_rating=("rating", "mean"),
            user_last_interaction=("timestamp", "max")
        )
        .reset_index()
    )

    user["days_since_last_interaction"] = (
        global_last - user["user_last_interaction"]
    ).dt.total_seconds() / 86400.0

    joined = train.merge(
        items[["movieId", "genres"]],
        on="movieId",
        how="left"
    )

    def preferred_genre(series):
        values = []
        for genres in series.dropna():
            values.extend(
                g for g in str(genres).split("|")
                if g and g != "(no genres listed)"
            )

        if not values:
            return "Unknown"

        return pd.Series(values).value_counts().index[0]

    preferred = (
        joined.groupby("userId")["genres"]
        .apply(preferred_genre)
        .reset_index()
        .rename(columns={"genres": "preferred_genre"})
    )

    user = user.merge(preferred, on="userId", how="left")

    # -------------------------------------------------
    # 2. ITEM FEATURES
    # Preserve the complete catalog using LEFT JOIN
    # -------------------------------------------------
    item_stats = (
        train.groupby("movieId")
        .agg(
            item_popularity=("userId", "count"),
            item_average_rating=("rating", "mean")
        )
        .reset_index()
    )

    item = items.merge(
        item_stats,
        on="movieId",
        how="left",
        validate="one_to_one"
    )

    item["item_popularity"] = (
        item["item_popularity"].fillna(0).astype(int)
    )

    item["has_training_interactions"] = (
        item["item_popularity"] > 0
    )

    # Keep item_average_rating as NaN for unrated movies.
    # Do not invent ratings for items without interactions.

    # -------------------------------------------------
    # 3. INTERACTION FEATURES
    # -------------------------------------------------
    interaction = train[
        ["userId", "movieId", "rating", "timestamp"]
    ].copy()

    interaction["days_from_global_latest"] = (
        global_last - interaction["timestamp"]
    ).dt.total_seconds() / 86400.0

    # -------------------------------------------------
    # 4. SAVE FEATURE DATASETS
    # -------------------------------------------------
    user.to_csv(
        FEATURES / "user_features.csv",
        index=False
    )

    item.to_csv(
        FEATURES / "item_features.csv",
        index=False
    )

    interaction.to_csv(
        FEATURES / "interaction_features.csv",
        index=False
    )

    # -------------------------------------------------
    # 5. FEATURE CATALOG
    # -------------------------------------------------
    catalog = [
        ["user_activity_count", "user",
         "Number of training interactions", "Integer",
         "Training interactions only"],

        ["user_average_rating", "user",
         "Mean rating given by user", "Float",
         "Training interactions only"],

        ["days_since_last_interaction", "user",
         "Days between user's last training event and global training cutoff",
         "Float", "Training cutoff"],

        ["preferred_genre", "user",
         "Most frequently interacted genre", "String",
         "Training interactions and item genres"],

        ["item_popularity", "item",
         "Number of training interactions for item", "Integer",
         "Training interactions only"],

        ["item_average_rating", "item",
         "Mean training rating for item", "Float",
         "Training interactions only"],

        ["genres", "item",
         "Movie genre metadata", "String",
         "Movie catalog"],

        ["has_training_interactions", "item",
         "Whether item has at least one training interaction",
         "Boolean", "Training interactions only"],

        ["rating", "interaction",
         "Explicit user rating", "Float",
         "Observed interaction"],

        ["days_from_global_latest", "interaction",
         "Days before latest training timestamp", "Float",
         "Training cutoff"]
    ]

    pd.DataFrame(
        catalog,
        columns=[
            "feature",
            "entity",
            "description",
            "data_type",
            "computation"
        ]
    ).to_csv(
        FEATURES / "feature_catalog.csv",
        index=False
    )

    print("FEATURE_BUILD_SUMMARY:", {
        "user_features": len(user),
        "item_features": len(item),
        "interaction_features": len(interaction),
        "catalog_items": len(items),
        "items_without_training_interactions": int(
            (~item["has_training_interactions"]).sum()
        ),
        "feature_catalog_entries": len(catalog)
    })

    return user, item, interaction


if __name__ == "__main__":
    build_features()
