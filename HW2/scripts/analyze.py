"""Compare measured errors and cold-start slices without fitting additional models."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.evaluate import scores
from src.data import ROOT, ACTIVITIES, load_tables


def main():
    report = json.loads((ROOT / "reports/results.json").read_text(encoding="utf-8"))
    tables = load_tables(ROOT / "data/processed/db")
    truth = pd.read_csv(ROOT / "reports/validation_truth.csv")
    cutoff = pd.Timestamp(report["config"]["validation_cutoff_utc"])
    known = tables["Users"].loc[tables["Users"]["first_seen_at"].lt(cutoff), "github_user_id"]
    visible = truth["UserID"].isin(known)
    analysis = {"groups": {}, "per_activity_delta_vs_copy_last_week": {}}
    for name, mask in [("previously_observed", visible), ("cold_start", ~visible)]:
        target = truth.loc[mask]
        group = {"users": len(target), "positive_labels": int(target[ACTIVITIES].to_numpy().sum()),
                 "positive_counts": {k: int(target[k].sum()) for k in ACTIVITIES}, "models": {}}
        if len(target):
            for model in [r["model"] for r in report["results"]]:
                prediction = pd.read_csv(ROOT / f"reports/validation_{model}.csv")
                prediction = prediction.loc[prediction["UserID"].isin(target["UserID"])]
                group["models"][model] = scores(prediction, target)
        analysis["groups"][name] = group
    by_model = {r["model"]: r for r in report["results"]}
    for activity in ACTIVITIES:
        analysis["per_activity_delta_vs_copy_last_week"][activity] = 100*(
            by_model["relational_temporal"]["per_event"][activity]
            - by_model["copy_last_week"]["per_event"][activity])
    analysis["pr_author_missing_pct"] = float(100*tables["PullRequests"]["author_id"].isna().mean())
    analysis["pr_open_time_missing_pct"] = float(100*tables["PullRequests"]["created_at"].isna().mean())
    analysis["note"] = "Subgroup results are descriptive, not used for model selection; empty-label F1 follows official code."
    (ROOT / "reports/analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(json.dumps({k: {"users":v["users"],"positive_labels":v["positive_labels"]}
                      for k,v in analysis["groups"].items()}))


if __name__ == "__main__":
    main()
