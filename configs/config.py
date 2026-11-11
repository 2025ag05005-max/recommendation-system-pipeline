from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
FEATURES = DATA / "features"
ERRORS = DATA / "errors"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
MLRUNS = ROOT / "mlruns"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RANDOM_SEED = 42
K = 5
RELEVANT_RATING = 4.0

for path in [RAW, PROCESSED, FEATURES, ERRORS, MODELS, REPORTS, LOGS]:
    path.mkdir(parents=True, exist_ok=True)
