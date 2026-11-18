"""Execute historical experiments and generate reproducible, official-format predictions."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from scripts.evaluate import scores, read_predictions
from src.data import (ACTIVITIES, ROOT, cohort_codes, copy_last_week, labels, load_tables,
                      make_snapshot, table_audit, validate_config, TABLES)
from src.model import ActivityModel, fit, predict, tensors, tune_thresholds


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def output_frame(codes, values):
    frame = pd.DataFrame(values.astype(np.int64), columns=ACTIVITIES)
    frame.insert(0, "UserID", np.asarray(codes, dtype=np.int64))
    return frame


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_summary(config):
    manifest = ROOT / "data/raw_manifest.jsonl"
    entries = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            entries[item["file"]] = item
    start, stop = pd.Timestamp(config["raw_start_utc"]), pd.Timestamp(config["raw_end_utc_exclusive"])
    expected = [f"{hour:%Y-%m-%d}-{hour.hour}.json.gz"
                for hour in pd.date_range(start, stop, freq="h", inclusive="left")]
    missing = [name for name in expected if entries.get(name, {}).get("status") == "missing"]
    failed = [name for name in expected if entries.get(name, {}).get("status") not in
              ("downloaded", "cached", "missing")]
    return {"requested_hours": len(expected), "available_hours": len(expected)-len(missing)-len(failed),
            "missing_hours": missing, "unaccounted_or_failed_hours": failed,
            "raw_start_utc": str(start), "raw_end_utc_exclusive": str(stop)}


def smoke(tables, config):
    # A mechanics-only check on the two-hour pilot. No performance claim is made.
    first = min(frame[TABLES[name]["time"]].min() for name, frame in tables.items() if len(frame))
    cutoff = first + pd.Timedelta(minutes=30)
    report = {"purpose": "real-data execution smoke test; target week is incomplete, no scores computed"}
    for temporal in [False, True]:
        snapshot = make_snapshot(tables, cutoff, temporal, max_nodes=2000)
        model = ActivityModel({k: v.shape[1] for k, v in snapshot.features.items()},
                              snapshot.edges, hidden=16, layers=2, relational=True)
        logits = model(*tensors(snapshot))
        target = torch.tensor(labels(tables, cutoff, snapshot.codes))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all()
                                                for p in model.parameters() if p.grad is not None):
            raise ValueError("Non-finite model output/gradient")
        report[str(temporal)] = {"shape": list(logits.shape), "finite_loss": float(loss.detach()),
                                 "audit": snapshot.audit}
    write_json(ROOT / "reports/smoke.json", report)
    print("SMOKE PASS: official tables -> time-safe graphs -> six logits -> finite backward pass", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--db", default="data/processed/db")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.epochs:
        config["epochs"] = args.epochs
    validate_config(config)
    tables = load_tables(ROOT / args.db)
    if args.smoke:
        smoke(tables, config)
        return
    if args.predict_only:
        name = "relational_temporal"
        metadata = json.loads((ROOT / f"artifacts/{name}.json").read_text(encoding="utf-8"))
        trained_config = metadata["config"]
        snapshot = make_snapshot(tables, trained_config["prediction_cutoff_utc"], True,
                                 trained_config["lookback_days"], trained_config["max_nodes_per_table"])
        dimensions = {k: v.shape[1] for k, v in snapshot.features.items()}
        model = ActivityModel(dimensions, snapshot.edges, trained_config["hidden_dim"],
                              trained_config["graph_layers"], True)
        model.load_state_dict(torch.load(ROOT / f"artifacts/{name}.pt", weights_only=True))
        test_users = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
        positions = pd.Index(snapshot.codes).get_indexer(test_users)
        if (positions < 0).any():
            raise ValueError("Test users outside cohort")
        values = (predict(model, snapshot) >= np.array(metadata["thresholds"]))[positions]
        destination = ROOT / "artifacts/output_reproduced.csv"
        output_frame(test_users, values).to_csv(destination, index=False)
        read_predictions(destination, test_users)
        original = ROOT / f"output_{trained_config['student_id']}.csv"
        equal = file_hash(original) == file_hash(destination)
        write_json(ROOT / "reports/reproducibility.json", {
            "identical_csv": equal, "original_sha256": file_hash(original),
            "reproduced_sha256": file_hash(destination)})
        if not equal:
            raise ValueError("Reloaded model did not reproduce submitted output")
        print("REPRODUCIBILITY PASS: reloaded model produces byte-identical CSV", flush=True)
        return
    started = time.monotonic()
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    coverage = manifest_summary(config)
    if coverage["unaccounted_or_failed_hours"]:
        raise ValueError("Complete the requested downloads before reporting experiments")
    codes = cohort_codes()
    test_users = pd.read_csv(ROOT / "submission/test_input.csv")["UserID"].to_numpy(dtype=np.int64)
    positions = pd.Index(codes).get_indexer(test_users)
    if (positions < 0).any():
        raise ValueError("Test list contains a user outside the official cohort")
    validate_time = config["validation_cutoff_utc"]
    tune_time = config["tune_cutoff_utc"]
    y_val = labels(tables, validate_time, codes)
    y_tune = labels(tables, tune_time, codes)
    truth = output_frame(test_users, y_val[positions])
    truth.to_csv(ROOT / "reports/validation_truth.csv", index=False)
    results = []
    baseline = output_frame(test_users, copy_last_week(tables, validate_time, codes)[positions])
    baseline.to_csv(ROOT / "reports/validation_copy_last_week.csv", index=False)
    baseline_score = scores(baseline, truth)
    results.append({"model": "copy_last_week", **baseline_score, "training_seconds": 0.0})
    write_json(ROOT / "reports/data_audit.json", {
        "coverage": coverage, "tables": table_audit(tables),
        "test_users": len(test_users), "cohort_users": len(codes),
        "validation_positive_counts": dict(zip(ACTIVITIES, y_val[positions].sum(0).astype(int).tolist())),
        "label_note": "Labels reconstruct observed activities only; missing GH Archive events are not real negatives.",
        "preprocess_sha256": file_hash(ROOT / "scripts/preprocess.py"),
        "official_commit": config["official_commit"]})
    variants = [("user_only", False, False), ("relational", True, False),
                ("relational_temporal", True, True)]
    for name, relational, temporal in variants:
        clock = time.monotonic()
        print(f"\nMODEL {name}", flush=True)
        def snapshot(value):
            return make_snapshot(tables, value, temporal, config["lookback_days"],
                                 config["max_nodes_per_table"])
        train = [snapshot(value) for value in config["train_cutoffs_utc"]]
        train_y = [labels(tables, value, codes) for value in config["train_cutoffs_utc"]]
        tuning, validation = snapshot(tune_time), snapshot(validate_time)
        dimensions = {k: v.shape[1] for k, v in train[0].features.items()}
        model, history = fit(train, train_y, config, relational)
        thresholds = tune_thresholds(predict(model, tuning)[positions], y_tune[positions])
        torch.save(model.state_dict(), ROOT / f"artifacts/{name}.pt")
        probabilities = predict(model, validation)
        val_prediction = output_frame(test_users, (probabilities >= thresholds)[positions])
        val_prediction.to_csv(ROOT / f"reports/validation_{name}.csv", index=False)
        metrics = scores(val_prediction, truth)
        elapsed = time.monotonic() - clock
        results.append({"model": name, **metrics, "training_seconds": elapsed,
                        "thresholds": thresholds.tolist()})
        write_json(ROOT / f"artifacts/{name}.json", {"config": config, "thresholds": thresholds.tolist(),
                                                   "loss": history, "validation": metrics})
        write_json(ROOT / f"reports/snapshots_{name}.json", {
            "train": [s.audit for s in train], "tuning": tuning.audit, "validation": validation.audit})
        final = snapshot(config["prediction_cutoff_utc"])
        final_values = (predict(model, final) >= thresholds)[positions]
        output_frame(test_users, final_values).to_csv(ROOT / f"reports/prediction_{name}.csv", index=False)
        print(f"{name}: macro={metrics['macro_f1']:.4f}, micro={metrics['micro_f1']:.4f}, "
              f"score={metrics['score']:.4f}, {elapsed:.1f}s", flush=True)
        del model, train, tuning, validation, final
        gc.collect()
    # Predeclared primary model; validation is for reporting, not choosing a winning model.
    primary = ROOT / "reports/prediction_relational_temporal.csv"
    destination = ROOT / f"output_{config['student_id']}.csv"
    destination.write_bytes(primary.read_bytes())
    read_predictions(destination, test_users)
    write_json(ROOT / "reports/results.json", {"config": config, "results": results,
               "primary_model": "relational_temporal", "total_seconds": time.monotonic()-started,
               "output_sha256": file_hash(destination), "prediction_data_end_utc": config["raw_end_utc_exclusive"],
               "limitations": ["Missing or incomplete GH Archive events affect labels.",
                   "PR/issue/review labels are reconstructed from table timestamps and may be incomplete.",
                   "The model predicts four days beyond the last available source data before the test week.",
                   "Graph node caps, if triggered, are recorded in snapshot audit files."]})
    rows = [{"model": r["model"], "macro_f1": r["macro_f1"], "micro_f1": r["micro_f1"],
             "score": r["score"], **r["per_event"]} for r in results]
    pd.DataFrame(rows).to_csv(ROOT / "reports/metrics.csv", index=False)
    print(f"\nPASS: {destination.name}, {len(test_users)} users, format verified", flush=True)


if __name__ == "__main__":
    main()
