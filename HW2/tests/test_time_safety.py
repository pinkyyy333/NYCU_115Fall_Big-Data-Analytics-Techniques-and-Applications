"""Behavioral checks for leakage, labels, temporal splits, and foreign-key visibility."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.data import (ROOT, TABLES, cohort_codes, labels, make_snapshot, official,
                      validate_config)


def fixture_tables():
    code = int(cohort_codes()[0])
    frames = {}
    for name, schema in TABLES.items():
        frames[name] = official.set_types(pd.DataFrame(columns=[c for c, _ in schema["columns"]]), schema)

    def put(name, rows):
        columns = [c for c, _ in TABLES[name]["columns"]]
        frames[name] = official.set_types(pd.DataFrame(rows, columns=columns), TABLES[name])

    put("Users", [{"user_id": 0, "github_user_id": code, "is_cohort": True,
                   "first_seen_at": "2026-10-01T00:00:00Z"}])
    put("Repositories", [{"repo_id": 0, "github_repo_id": 123456789012345,
                          "first_seen_at": "2026-10-01T00:00:00Z"}])
    put("PullRequests", [{"pr_id": 0, "github_pr_id": 123456789012346,
                         "repo_id": 0, "author_id": 0, "merged_by_id": 0,
                         "created_at": "2026-10-02T00:00:00Z",
                         "first_seen_at": "2026-10-10T00:00:00Z",
                         "merged_at": "2026-10-20T00:00:00Z"}])
    put("Pushes", [{"push_id": i, "github_push_id": 123456789012350+i,
                    "user_id": 0, "repo_id": 0, "created_at": value}
                   for i, value in enumerate(["2026-10-11T00:00:00Z", "2026-10-18T00:00:00Z"])])
    return frames, code


def test_late_discovered_pr_not_available_early():
    tables, code = fixture_tables()
    early = make_snapshot(tables, "2026-10-09", False)
    assert early.audit["PullRequests"]["retained_nodes"] == 0
    assert labels(tables, "2026-10-01", np.array([code]))[0, 0] == 0


def test_target_window_excludes_right_boundary():
    tables, code = fixture_tables()
    assert labels(tables, "2026-10-11", np.array([code]))[0, 5] == 1
    # At 00:00 Oct 18 the second push belongs to the NEXT window.
    assert labels(tables, "2026-10-12", np.array([code]), horizon_days=6)[0, 5] == 0


def test_future_merge_identity_is_hidden():
    tables, _ = fixture_tables()
    early = make_snapshot(tables, "2026-10-15", False)
    later = make_snapshot(tables, "2026-10-21", False)
    key = ("PullRequests", "merged_by_id", "Users")
    assert early.edges[key].shape[1] == 0
    assert later.edges[key].shape[1] == 1


def test_future_rows_do_not_change_past_features():
    tables, _ = fixture_tables()
    a = make_snapshot(tables, "2026-10-15", True)
    altered = copy.deepcopy(tables)
    altered["Pushes"].loc[1, "created_at"] = pd.Timestamp("2026-11-01")
    b = make_snapshot(altered, "2026-10-15", True)
    for name in TABLES:
        np.testing.assert_array_equal(a.features[name], b.features[name])
    for key in a.edges:
        np.testing.assert_array_equal(a.edges[key], b.edges[key])


def test_overlapping_time_splits_are_rejected():
    config = json.loads((ROOT / "config/experiment.json").read_text(encoding="utf-8"))
    validate_config(config)
    config["train_cutoffs_utc"] = [config["tune_cutoff_utc"]]
    with pytest.raises(ValueError, match="Training labels overlap"):
        validate_config(config)
