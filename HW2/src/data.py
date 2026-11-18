"""Snapshots and observable activity labels from the eight official tables only."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("official_preprocess", ROOT / "scripts/preprocess.py")
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)
TABLES = official.TABLES
ACTIVITIES = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
EVENT_SPECS = [
    ("PullRequests", "author_id", "created_at"),
    ("Issues", "author_id", "created_at"),
    ("IssueComments", "author_id", "created_at"),
    ("PRReviews", "reviewer_id", "submitted_at"),
    ("PRReviewComments", "author_id", "created_at"),
    ("Pushes", "user_id", "created_at"),
]


def validate_config(config):
    horizon = pd.Timedelta(days=config["horizon_days"])
    tune = pd.Timestamp(config["tune_cutoff_utc"])
    validation = pd.Timestamp(config["validation_cutoff_utc"])
    end = pd.Timestamp(config["raw_end_utc_exclusive"])
    prediction = pd.Timestamp(config["prediction_cutoff_utc"])
    for value in config["train_cutoffs_utc"]:
        if pd.Timestamp(value) + horizon > tune:
            raise ValueError("Training labels overlap the tuning prediction time")
    if tune + horizon > validation:
        raise ValueError("Tuning labels overlap final validation")
    if validation + horizon > end:
        raise ValueError("Validation target week has not been fully downloaded")
    if end > prediction:
        raise ValueError("Downloaded data extends into the final test week")


def load_tables(folder):
    result = {}
    for name, schema in TABLES.items():
        frame = pd.read_parquet(Path(folder) / f"{name}.parquet")
        for column, kind in schema["columns"]:
            if kind == "time":
                frame[column] = pd.to_datetime(frame[column])
        result[name] = frame
    return result


def cohort_codes():
    return np.array(sorted(official.load_cohort(ROOT / "scripts/cohort.txt")), dtype=np.int64)


def activity_rows(tables, index, known_before):
    """Rows of an activity, constrained by when the row becomes observable."""
    name, actor, timestamp = EVENT_SPECS[index]
    frame = tables[name]
    mask = frame[TABLES[name]["time"]].lt(pd.Timestamp(known_before))
    mask &= frame[timestamp].notna() & frame[actor].notna()
    if name == "Issues":
        mask &= ~frame["is_pull_request"].fillna(False)
    return frame.loc[mask], actor, timestamp


def labels(tables, cutoff, codes, horizon_days=7):
    """Observed positives, NOT a claim that an unobserved event never happened."""
    start = pd.Timestamp(cutoff)
    stop = start + pd.Timedelta(days=horizon_days)
    code_positions = {int(code): i for i, code in enumerate(codes)}
    pk_to_code = tables["Users"].set_index("user_id")["github_user_id"]
    target = np.zeros((len(codes), 6), dtype=np.float32)
    for k in range(6):
        rows, actor, timestamp = activity_rows(tables, k, stop)
        rows = rows.loc[rows[timestamp].ge(start) & rows[timestamp].lt(stop)]
        values = rows[actor].map(pk_to_code).map(code_positions).dropna().astype(int)
        target[values.to_numpy(), k] = 1
    return target


def copy_last_week(tables, cutoff, codes):
    start = pd.Timestamp(cutoff) - pd.Timedelta(days=7)
    return labels(tables, start, codes)


@dataclass
class Snapshot:
    cutoff: str
    features: dict
    edges: dict
    cohort_nodes: np.ndarray
    codes: np.ndarray
    audit: dict


def _numeric(frame, name):
    if name not in frame:
        return np.zeros(len(frame), dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype=np.float32)


def make_snapshot(tables, cutoff, temporal, lookback_days=14, max_nodes=40000):
    t = pd.Timestamp(cutoff)
    codes = cohort_codes()
    frames, maps, features, audit = {}, {}, {}, {}
    windows = [1, 3, 7, lookback_days] if temporal else [lookback_days]
    # Six base attributes; activity summaries are populated only on users/repositories.
    dim = 6 + 6 * len(windows) + (13 if temporal else 0)
    for name, schema in TABLES.items():
        eligible = tables[name].loc[tables[name][schema["time"]].lt(t)]
        total = len(eligible)
        if name != "Users" and len(eligible) > max_nodes:
            eligible = eligible.sort_values(schema["time"], kind="stable").tail(max_nodes)
        frame = eligible.copy().reset_index(drop=True)
        frames[name] = frame
        maps[name] = pd.Series(np.arange(len(frame)), index=frame[schema["pk"]])
        n = len(frame)
        x = np.zeros((n, dim), dtype=np.float32)
        x[:, 0] = 1
        x[:, 1] = _numeric(frame, "is_cohort")
        x[:, 2] = _numeric(frame, "is_pull_request")
        for j, col in enumerate(["title", "body"], 3):
            if col in frame:
                x[:, j] = np.log1p(frame[col].fillna("").str.len().to_numpy()) / 5
        x[:, 5] = np.log1p(_numeric(frame, "num_labels"))
        if temporal:
            x[:, -13] = np.minimum((t - frame[schema["time"]]).dt.total_seconds().to_numpy()
                                    / 86400, 90) / 30
        features[name] = x
        audit[name] = {"eligible_nodes": total, "retained_nodes": n,
                       "capped": total != n, "latest_row_time": str(frame[schema["time"]].max())}

    user_codes = frames["Users"]["github_user_id"].astype("int64").to_numpy()
    code_to_node = {int(code): i for i, code in enumerate(user_codes)}
    missing = [int(code) for code in codes if int(code) not in code_to_node]
    old_n = len(user_codes)
    code_to_node.update({code: old_n + i for i, code in enumerate(missing)})
    if missing:
        features["Users"] = np.concatenate(
            [features["Users"], np.zeros((len(missing), dim), dtype=np.float32)])
    cohort_nodes = np.array([code_to_node[int(code)] for code in codes], dtype=np.int64)
    audit["cohort_unseen"] = len(missing)

    # Features use event times AND row availability; late PR/issue/review discoveries cannot leak.
    for k in range(6):
        rows, actor, timestamp = activity_rows(tables, k, t)
        rows = rows.loc[rows[timestamp].lt(t) & rows[timestamp].ge(t - pd.Timedelta(days=lookback_days))]
        ages = (t - rows[timestamp]).dt.total_seconds().to_numpy() / 86400
        for node_type, foreign_key in [("Users", actor), ("Repositories", "repo_id")]:
            positions = rows[foreign_key].map(maps[node_type])
            good = positions.notna().to_numpy()
            idx = positions.loc[good].to_numpy(dtype=np.int64)
            age = ages[good]
            n = len(features[node_type])
            for j, days in enumerate(windows):
                count = np.bincount(idx[age <= days], minlength=n)
                features[node_type][:, 6 + j * 6 + k] = np.log1p(count) / 3
            if temporal:
                decay = np.bincount(idx, weights=np.exp(-age / 3), minlength=n)
                features[node_type][:, -12 + k] = np.log1p(decay) / 3
                recency = np.full(n, lookback_days, dtype=np.float32)
                np.minimum.at(recency, idx, age.astype(np.float32))
                features[node_type][:, -6 + k] = 1 - recency / lookback_days

    edges = {}
    for name, schema in TABLES.items():
        frame = frames[name]
        for fk, reference in schema["fks"].items():
            positions = frame[fk].map(maps[reference])
            valid = positions.notna()
            if name == "PullRequests" and fk == "merged_by_id":
                valid &= frame["merged_at"].notna() & frame["merged_at"].lt(t)
            src = np.flatnonzero(valid.to_numpy()).astype(np.int64)
            dst = positions.loc[valid].to_numpy(dtype=np.int64)
            key = (name, fk, reference)
            edges[key] = np.stack([src, dst])
            edges[(reference, f"rev_{name}_{fk}", name)] = np.stack([dst, src])
    audit["edge_count"] = int(sum(e.shape[1] for e in edges.values()))
    audit["temporal"] = temporal
    audit["feature_dimension"] = dim
    return Snapshot(str(t), features, edges, cohort_nodes, codes, audit)


def table_audit(tables):
    summary = {}
    for name, frame in tables.items():
        time = TABLES[name]["time"]
        summary[name] = {"rows": len(frame), "row_time_start": str(frame[time].min()),
                         "row_time_end": str(frame[time].max()),
                         "null_fraction": {col: float(frame[col].isna().mean()) for col in frame}}
        for fk, reference in TABLES[name]["fks"].items():
            values = frame[fk].dropna()
            if not values.isin(tables[reference][TABLES[reference]["pk"]]).all():
                raise ValueError(f"Broken foreign key {name}.{fk}")
    return summary
