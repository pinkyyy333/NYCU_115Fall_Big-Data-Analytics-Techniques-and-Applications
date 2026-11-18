# HW2: Predicting GitHub User Activity

This public submission contains the source code, configuration and dependencies needed to
reproduce data preparation, features, model training and six binary activity predictions.
Data, model weights, per-user outputs, slides, speaker notes and supporting analysis stay local.

## Environment

Verified with Python 3.13 on Windows and CPU PyTorch. From this directory in PowerShell:

```powershell
uv --cache-dir .cache/uv venv .venv --python 3.13
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe --no-deps torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv --cache-dir .cache/uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

`requirements.txt` pins the model's runtime dependencies. The first install selects the CPU
wheel; the second installs its complete locked dependencies from PyPI.

## Run the complete pipeline

```powershell
powershell -File scripts/run_all.ps1
```

The script fetches the official anonymized cohort/test lists from the pinned course commit,
verifies their SHA-256 hashes, downloads GH Archive hours, runs the unchanged official
preprocessing script, trains the models, predicts activities and checks output/reload consistency.
The official lists are downloaded locally rather than republished here.

The configured raw period is Oct 11 00:00–Nov 18 00:00, 2026 (Taiwan time): 912 hourly files,
about 5.82 GiB in the verified download. Downloads resume at completed-file boundaries.
The downloader has a 60 GiB limit; keep additional disk space for generated tables and outputs.
With a complete local raw download, use `powershell -File scripts/run_all.ps1 -SkipDownload`.

Individual commands:

```powershell
.venv/Scripts/python.exe scripts/fetch_official_inputs.py
.venv/Scripts/python.exe scripts/download.py --start 2026-10-10T16:00:00 --end 2026-11-17T16:00:00 --out data/raw --workers 6 --max-gb 60
.venv/Scripts/python.exe scripts/preprocess.py --raw data/raw --out data/processed
.venv/Scripts/python.exe run.py
.venv/Scripts/python.exe run.py --predict-only
.venv/Scripts/python.exe scripts/evaluate.py --pred output_YOUR_STUDENT_ID.csv --input submission/test_input.csv
```

## Submission filename and output

The public configuration uses `YOUR_STUDENT_ID` for the filename. To set the actual ID
without publishing it, create a Git-ignored local configuration:

```powershell
Copy-Item config/experiment.json config/experiment.local.json
# Edit student_id in config/experiment.local.json locally.
powershell -File scripts/run_all.ps1 -Config config/experiment.local.json
```

Filename metadata does not affect prediction values. The CSV has one row per official test
user, exactly the following header and only integer 0/1 predictions:

```text
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

CSV/PDF/MP4 submission files are submitted to E3, not included in this public code directory.

## Reproduction settings

`config/experiment.json` records the seed, model settings and UTC timestamps. The implementation
uses only the eight tables produced by the official script. The primary model is a two-layer,
32-dimensional heterogeneous GNN with relation-specific neighbor means and temporal features.
It also runs copy-last-week, user-only and non-temporal relational baselines.

Taiwan-time prediction origins are Oct 25 (training), Nov 1 (threshold tuning), Nov 8 (historical
validation) and Nov 22 (official prediction). Each label window is the next seven days.
Features require row availability before the origin; late merge identities are time-masked.
Final validation is not used to select the model or thresholds. Source gaps and delayed row
discovery limit observed labels; no future official-test labels are used.

## Data Source

Data source: [GH Archive](https://www.gharchive.org/), used for this course's educational assignment.
The unchanged preprocessing/evaluation scripts and official input lists originate from
[NYCU BigData2026 Assignment 2](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2),
commit `af9f335301d9badb482dfafa10c4534eabac845a`.
