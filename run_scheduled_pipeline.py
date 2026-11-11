"""Simple recurring scheduler using only the Python standard library."""
import argparse
import time
from datetime import datetime
from src.orchestration.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run recommendation pipeline on a recurring schedule.")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    parser.add_argument("--run-immediately", action="store_true")
    args = parser.parse_args()
    if args.interval_hours <= 0:
        parser.error("--interval-hours must be greater than zero")
    interval = args.interval_hours * 3600
    print(f"Scheduler started. Interval: {args.interval_hours} hour(s).")
    if args.run_immediately:
        print("Immediate run:", datetime.now().isoformat(timespec="seconds"))
        run_pipeline()
    while True:
        next_run = datetime.now().timestamp() + interval
        print(f"Next run at: {datetime.fromtimestamp(next_run).isoformat(timespec='seconds')}")
        time.sleep(interval)
        print("Scheduled run:", datetime.now().isoformat(timespec="seconds"))
        run_pipeline()

if __name__ == "__main__":
    main()
