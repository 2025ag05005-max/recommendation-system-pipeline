import pandas as pd
from configs.config import PROCESSED, FEATURES

def transform():
    train = pd.read_csv(PROCESSED / "train_interactions.csv", parse_dates=["timestamp"])
    items = pd.read_csv(PROCESSED / "items_clean.csv")
    user_activity = train.groupby("userId").agg(
        activity_count=("movieId","count"),
        user_avg_rating=("rating","mean"),
        last_interaction=("timestamp","max")
    ).reset_index()
    item_popularity = train.groupby("movieId").agg(
        interaction_count=("userId","count"),
        item_avg_rating=("rating","mean")
    ).reset_index()
    item_popularity = items[["movieId","title","genres"]].merge(
        item_popularity, on="movieId", how="left"
    ).fillna({"interaction_count":0,"item_avg_rating":0})
    matrix = train.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0)
    user_activity.to_csv(FEATURES / "user_activity_summary.csv", index=False)
    item_popularity.to_csv(FEATURES / "item_popularity.csv", index=False)
    matrix.to_csv(FEATURES / "user_item_matrix.csv")
    print({
        "user_activity_rows": len(user_activity),
        "item_popularity_rows": len(item_popularity),
        "user_item_matrix_shape": matrix.shape
    })
    return user_activity, item_popularity, matrix

if __name__ == "__main__":
    transform()
