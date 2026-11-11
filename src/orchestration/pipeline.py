import json
import time
import traceback
from datetime import datetime, timezone
import mlflow
from configs.config import LOGS, MLRUNS
from src.ingestion.ingest import ingest
from src.validation.validate import validate
from src.preparation.prepare import prepare
from src.transformation.transform import transform
from src.features.build_features import build_features
from src.training.models import train_models
from src.evaluation.evaluate import evaluate

def run_pipeline():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    tasks = [
        ("ingestion", ingest, []),
        ("validation", validate, ["ingestion"]),
        ("preparation", prepare, ["validation"]),
        ("transformation", transform, ["preparation"]),
        ("feature_management", build_features, ["transformation"]),
        ("model_training", train_models, ["feature_management"]),
        ("model_evaluation", evaluate, ["model_training"])
    ]
    states = {}
    started = datetime.now(timezone.utc).isoformat()
    for name, fn, dependencies in tasks:
        failed_deps = [d for d in dependencies if states.get(d) != "SUCCESS"]
        if failed_deps:
            states[name] = "SKIPPED"
            raise RuntimeError(f"Task {name} skipped because dependencies failed: {failed_deps}")
        t0 = time.time()
        print(f"\n========== START: {name} ==========")
        try:
            fn()
            states[name] = "SUCCESS"
            print(f"========== SUCCESS: {name} ({time.time()-t0:.1f}s) ==========")
        except Exception:
            states[name] = "FAILED"
            error_path = LOGS / f"{name}_error.txt"
            error_path.write_text(traceback.format_exc())
            summary = {
                "status": "FAILED",
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "tasks": states,
                "failed_task": name,
                "error_log": str(error_path)
            }
            (LOGS / "pipeline_run.json").write_text(json.dumps(summary, indent=2))
            print("\nPIPELINE_RUN_SUMMARY:", json.dumps(summary, indent=2))
            raise
    summary = {
        "status": "SUCCESS" if all(v=="SUCCESS" for v in states.values()) else "FAILED",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": states
    }
    (LOGS / "pipeline_run.json").write_text(json.dumps(summary,indent=2))
    print("\nPIPELINE_RUN_SUMMARY:",json.dumps(summary,indent=2))
    return summary

if __name__ == "__main__":
    run_pipeline()
