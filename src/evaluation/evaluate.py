import json
import math
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
from configs.config import PROCESSED, REPORTS, MODELS, K, RELEVANT_RATING, MLRUNS


def ranking_metrics(model, train, test, items, k=K):
    item_genres = items.set_index("movieId")["genres"].fillna("").to_dict()
    train_seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    relevant = test[test["rating"] >= RELEVANT_RATING].groupby("userId")["movieId"].apply(set).to_dict()
    eligible = sorted(relevant.keys())
    precisions, recalls, ndcgs = [], [], []
    recommended_all = set()
    diversity_values = []
    catalog = set(items["movieId"].astype(int))
    for user_id in eligible:
        recs = model.recommend(int(user_id), train_seen.get(user_id,set()), k)
        recommended_all.update(recs)
        hits = [1 if r in relevant[user_id] else 0 for r in recs]
        precisions.append(sum(hits)/k)
        recalls.append(sum(hits)/len(relevant[user_id]))
        dcg = sum(hit/math.log2(i+2) for i,hit in enumerate(hits))
        ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant[user_id]),k)))
        ndcgs.append(dcg/ideal if ideal else 0.0)
        # Intra-list genre diversity: 1 - mean pairwise Jaccard similarity.
        pair_distances = []
        for i in range(len(recs)):
            gi = set(item_genres.get(recs[i],"").split("|")) - {"", "(no genres listed)"}
            for j in range(i+1,len(recs)):
                gj = set(item_genres.get(recs[j],"").split("|")) - {"", "(no genres listed)"}
                union = gi | gj
                similarity = len(gi & gj)/len(union) if union else 0.0
                pair_distances.append(1-similarity)
        diversity_values.append(float(np.mean(pair_distances)) if pair_distances else 0.0)

    # Rating prediction metrics over the held-out explicit ratings.
    y_true = test["rating"].astype(float).tolist()
    y_pred = [model.predict_rating(int(r.userId),int(r.movieId)) for r in test.itertuples()]
    rmse = float(np.sqrt(mean_squared_error(y_true,y_pred)))
    mae = float(mean_absolute_error(y_true,y_pred))
    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": float(len(recommended_all)/len(catalog)) if catalog else 0.0,
        "diversity": float(np.mean(diversity_values)) if diversity_values else 0.0,
        "rmse": rmse, "mae": mae,
        "eligible_users": int(len(eligible)),
        "evaluated_test_ratings": int(len(test))
    }

def evaluate():
    train = pd.read_csv(PROCESSED / "train_interactions.csv")
    test = pd.read_csv(PROCESSED / "test_interactions.csv")
    items = pd.read_csv(PROCESSED / "items_clean.csv")
    popularity = joblib.load(MODELS / 'popularity_model.joblib')
    item_cf = joblib.load(MODELS / 'item_cf_model.joblib')

    results = {}
    for name, model in [("Popularity",popularity),("ItemBasedCollaborativeFiltering",item_cf)]:
        metrics = ranking_metrics(model,train,test,items)
        results[name] = metrics
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model",name)
            mlflow.log_param("k",K)
            mlflow.log_param("relevant_rating_threshold",RELEVANT_RATING)
            mlflow.log_param("train_interactions",len(train))
            mlflow.log_param("test_interactions",len(test))
            for key,value in metrics.items():
                mlflow.log_metric(key,float(value))
            mlflow.log_artifact(str(REPORTS / "validation_report.json"))
            model_file = MODELS / ("popularity_model.joblib" if name == "Popularity" else "item_cf_model.joblib")
            mlflow.log_artifact(str(model_file))
    pd.DataFrame(results).T.to_csv(
        REPORTS / "model_comparison.csv",
        index=True,
        index_label="model"
    )
    (REPORTS / "evaluation_results.json").write_text(json.dumps(results,indent=2))
    print("MODEL_EVALUATION:",json.dumps(results,indent=2))
    return results

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    evaluate()
