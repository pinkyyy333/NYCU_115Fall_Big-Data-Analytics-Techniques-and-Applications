# HW2: Predicting GitHub User Activity

Predict six binary activities per user for Nov 22–29, 2026 (Taiwan time):
`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`.

This public directory contains the code needed to reproduce preprocessing, features,
training, historical evaluation and final predictions. Personal slides, student identifiers,
per-user outputs, downloaded data and model weights are kept out of version control.

## Environment

Verified with Python 3.13 and CPU PyTorch. From this directory in Windows PowerShell:

```powershell
uv --cache-dir .cache/uv venv .venv --python 3.13
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
```

`requirements.txt` lists direct dependencies; `requirements-lock.txt` pins the verified
core environment. Presentation-generation packages are not required.
The first install selects the CPU wheel; the second installs all locked dependencies from PyPI.

## Reproduce the pipeline

```powershell
powershell -File scripts/run_all.ps1
```

This command:

1. Fetches the official anonymized cohort/test lists from a pinned upstream commit,
   checking their SHA-256 hashes. These lists are downloaded locally, not republished here.
2. Downloads the configured GH Archive hours, retaining original bytes and a checksum manifest.
3. Runs the unchanged official `scripts/preprocess.py` to generate eight anonymized tables.
4. Checks temporal safety, trains all neural variants and evaluates the held-out week.
5. Generates predictions, checks the official format and verifies byte-identical model reload.
6. Writes local aggregate/error analysis.

The requested period is Oct 11 00:00–Nov 18 00:00 Taiwan time (912 hours).
The observed download size was about 5.82 GiB. Keep additional free space for tables and outputs.
Downloads resume at completed-file boundaries. With a complete local raw download, use
`powershell -File scripts/run_all.ps1 -SkipDownload`.

The default configuration uses generic author/filename placeholders. To set a personal
submission filename locally, copy it to a Git-ignored configuration:

```powershell
Copy-Item config/experiment.json config/experiment.local.json
# Edit student_id in config/experiment.local.json locally.
powershell -File scripts/run_all.ps1 -Config config/experiment.local.json
```

Author/filename metadata does not affect features, training or prediction values.

Individual steps:

```powershell
.venv/Scripts/python.exe scripts/fetch_official_inputs.py
.venv/Scripts/python.exe scripts/download.py --start 2026-10-10T16:00:00 --end 2026-11-17T16:00:00 --out data/raw --workers 6 --max-gb 60
.venv/Scripts/python.exe scripts/preprocess.py --raw data/raw --out data/processed
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
.venv/Scripts/python.exe run.py
.venv/Scripts/python.exe run.py --predict-only
.venv/Scripts/python.exe scripts/evaluate.py --pred output_YOUR_STUDENT_ID.csv --input submission/test_input.csv
```

The generated CSV has one row per official test user and exactly:

```text
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

All predictions are integers 0/1. Generated reports and predictions remain Git-ignored.

## Method

- Baselines: copy-last-week and a user-only neural model.
- Relational model: eight node types, all 20 schema foreign keys and their reverse edges,
  two heterogeneous message-passing layers, hidden dimension 32 and six sigmoid outputs.
- Each relation averages neighbor embeddings before a learned relation-specific transform.
  Self/neighbor updates use LayerNorm and ReLU.
- Non-temporal inputs: safe attributes and 14-day activity counts.
- Temporal inputs: 1/3/7/14-day counts, exponential decay, recency and row age (43 dimensions).
- Training: weighted BCE, positive-weight cap 15, AdamW, learning rate 0.003,
  weight decay 0.0001, 45 epochs, seed 42, deterministic CPU execution with four threads.
- All 9,000 official cohort users can train. Tuning/validation use the fixed 4,465-user test list.
  Unseen users receive zero observation features, not invented event histories.
- The primary method is declared in advance as relational + temporal.

See [SCHEMA.md](SCHEMA.md) and the [editable schema](docs/schema.drawio).
Only the eight tables built by the official preprocessing script are model inputs.

## Time split and leakage controls

| Purpose | Prediction origin (Taiwan time) | Seven-day target |
|---|---|---|
| Train | Oct 25, 00:00 | Oct 25–Nov 1 |
| Tune per-activity thresholds | Nov 1, 00:00 | Nov 1–8 |
| Final historical validation | Nov 8, 00:00 | Nov 8–15 |
| Official prediction | Nov 22, 00:00 | Nov 22–29 |

Features require `row_time < t`; event summaries also require `event_time < t`.
PRs/issues/reviews use `first_seen_at` for row availability. Merger edges require
`merged_at < t`; closed/dismissed states and latest usernames are excluded as inputs.
No training or tuning label window overlaps the next prediction origin.
The final historical validation week is not used for choosing a model or thresholds.

## Recorded historical results

| Method | Macro-F1 | Micro-F1 | Score |
|---|---:|---:|---:|
| Copy last week | 33.56% | 33.95% | 33.76% |
| User only | 29.94% | 30.71% | 30.33% |
| Relational | 33.57% | 33.77% | 33.67% |
| Relational + temporal | 34.34% | 35.01% | 34.68% |

`Score = (macro-F1 + micro-F1) / 2`. [Aggregate metrics](docs/metrics.csv) include F1 per activity.
These are observed-label results on the Nov 8–15 historical week, not future official-test scores.
The saved primary model reload was checked to produce a byte-identical prediction CSV.
The verified Windows CSV checksum is:

```text
SHA-256: b0d84d0073b48a600a6755116162f0c003d60f816bcdc75399fc11bcc101b809
```

## Limitations

- Missing GH Archive events and missing opening/author fields affect labels. About 68.14%
  of PR author/opening-time values are missing in this downloaded period.
- A missing observation is not guaranteed real inactivity; non-cohort histories are incomplete.
- Conservative row-time filtering can omit retrospectively discovered activity.
- Data ends four days before the official prediction origin.
- This is one historical split and one seed. Model capacities differ, and temporal features
  are tested as a bundle; the comparison does not isolate each feature's causal effect.
- The official evaluator sets an activity's F1 to 1 if both truth and prediction have no positives.
- Any applied graph-node caps are recorded in local snapshot audit files.

## Data Source

Data source: [GH Archive](https://www.gharchive.org/), used for this course's educational assignment.
The official preprocessing/evaluation scripts and schema originate from
[NYCU BigData2026 Assignment 2](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2),
commit `af9f335301d9badb482dfafa10c4534eabac845a`.
Official cohort/test inputs are retrieved from that same pinned commit.
