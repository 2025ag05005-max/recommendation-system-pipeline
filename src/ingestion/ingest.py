import zipfile
from pathlib import Path
import requests
import pandas as pd
from configs.config import RAW, MOVIELENS_URL

def ingest():
    """Download MovieLens and create explicit users, items, and interactions CSVs."""
    zip_path = RAW / "ml-latest-small.zip"
    if not zip_path.exists():
        response = requests.get(MOVIELENS_URL, timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

    extract_dir = RAW / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    base = extract_dir / "ml-latest-small"
    ratings = pd.read_csv(base / "ratings.csv")
    movies = pd.read_csv(base / "movies.csv")

    # MovieLens does not provide demographic user profiles. The users table
    # therefore contains observed user identifiers and a documented source field.
    users = pd.DataFrame({"userId": sorted(ratings["userId"].dropna().unique())})
    users["user_source"] = "MovieLens ratings.csv"

    users.to_csv(RAW / "users.csv", index=False)
    movies.to_csv(RAW / "items.csv", index=False)
    ratings.to_csv(RAW / "interactions.csv", index=False)

    summary = {
        "users": len(users), "items": len(movies), "interactions": len(ratings),
        "dataset": "MovieLens Latest Small", "source_url": MOVIELENS_URL
    }
    print("INGESTION_SUMMARY:", summary)
    return summary

if __name__ == "__main__":
    ingest()
