import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from configs.config import PROCESSED, MODELS

class PopularityRecommender:
    def fit(self, train, items):
        stats = train.groupby("movieId").agg(
            count=("rating","count"), mean_rating=("rating","mean")
        )
        self.ranking = items[["movieId","title","genres"]].merge(stats, on="movieId", how="left")
        self.ranking["count"] = self.ranking["count"].fillna(0)
        self.ranking["mean_rating"] = self.ranking["mean_rating"].fillna(train["rating"].mean())
        self.ranking["score"] = self.ranking["count"] * self.ranking["mean_rating"]
        self.global_mean = float(train["rating"].mean())
        return self

    def recommend(self, user_id, seen, k=5):
        return self.ranking[~self.ranking["movieId"].isin(seen)].sort_values(
            ["score","count","movieId"], ascending=[False,False,True]
        )["movieId"].head(k).tolist()

    def predict_rating(self, user_id, movie_id):
        row = self.ranking[self.ranking.movieId == movie_id]
        return float(row["mean_rating"].iloc[0]) if len(row) else self.global_mean

class ItemCFRecommender:
    def fit(self, train, items):
        self.items = items.copy()
        self.user_means = train.groupby("userId")["rating"].mean().to_dict()
        self.global_mean = float(train["rating"].mean())
        self.user_item = train.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0)
        self.movie_ids = self.user_item.columns.astype(int).tolist()
        self.user_ids = self.user_item.index.astype(int).tolist()
        self.matrix = csr_matrix(self.user_item.values)
        self.similarity = cosine_similarity(self.matrix.T, dense_output=False)
        self.movie_index = {m:i for i,m in enumerate(self.movie_ids)}
        self.user_index = {u:i for i,u in enumerate(self.user_ids)}
        return self

    def _scores(self, user_id):
        if user_id not in self.user_index:
            return {}
        row = self.user_item.loc[user_id].values.astype(float)
        scores = self.similarity @ row
        denom = np.abs(self.similarity) @ (row != 0).astype(float)
        scores = np.divide(scores, denom, out=np.zeros_like(scores), where=denom > 0)
        return {m:float(scores[i]) for i,m in enumerate(self.movie_ids)}

    def recommend(self, user_id, seen, k=5):
        scores = self._scores(user_id)
        candidates = [(m,s) for m,s in scores.items() if m not in seen]
        candidates.sort(key=lambda x:(-x[1],x[0]))
        return [m for m,_ in candidates[:k]]

    def predict_rating(self, user_id, movie_id):
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return self.global_mean
        u = self.user_item.loc[user_id].values.astype(float)
        target_idx = self.movie_index[movie_id]
        sims = self.similarity[target_idx].toarray().ravel()
        rated = u > 0
        rated[target_idx] = False
        weights = sims[rated]
        values = u[rated]
        if len(values) == 0 or np.abs(weights).sum() == 0:
            return float(self.user_means.get(user_id, self.global_mean))
        pred = float(np.dot(weights, values) / np.abs(weights).sum())
        return float(np.clip(pred, 0.5, 5.0))

def train_models():
    train = pd.read_csv(PROCESSED / "train_interactions.csv")
    items = pd.read_csv(PROCESSED / "items_clean.csv")
    popularity = PopularityRecommender().fit(train, items)
    item_cf = ItemCFRecommender().fit(train, items)
    joblib.dump(popularity, MODELS / "popularity_model.joblib")
    joblib.dump(item_cf, MODELS / "item_cf_model.joblib")
    return popularity, item_cf, train, items


if __name__ == "__main__":
    popularity, item_cf, train, items = train_models()
    print({
        "status": "SUCCESS",
        "models": ["PopularityRecommender", "ItemCFRecommender"],
        "training_interactions": len(train),
        "catalog_items": len(items)
    })
