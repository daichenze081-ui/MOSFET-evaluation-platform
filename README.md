# MOSFET Device Analytics

A measured-data workflow for MOSFET assessment, compact-model fitting, and
traceable prediction reports. Valid measurements determine the device
conclusion; predictions cannot override them.

## Diagnose metrics locally

```powershell
python -m examples.diagnosis_demo
python -m mosfet_platform.cli diagnose --request outputs/local/request.json
```

The diagnostic workflow checks only the supplied Spec rules, preserves valid
evidence when other metrics are missing, and reports per-metric exceedance,
co-occurrence, and geometry groups separately for each data source. Each run
saves JSON, CSV, and an HTML report. The callable `mosfet_platform.api.diagnose`
service and request JSON Schema provide the boundary for a future agent.
See [local diagnostic inputs and results](docs/diagnosis.md). Model completion
is not performed by this entry point; unavailable metrics remain explicit gaps.

## Run the complete workflow

```powershell
python -m pip install -e ".[dev]"
python -m examples.platform_demo
```

Open `outputs/platform/result.html`. This is the single result entry point.
The example uses synthetic curves, conditions, and limits generated at runtime;
it is software demonstration data, not engineering qualification evidence.

The workflow validates measurements, stores accepted data in SQLite, trains
when the curve dataset changes, and reports measured decisions alongside model
errors. Valid PASS and FAIL measurements are both retained. Reimporting the
same content does not duplicate observations or retrain the model.

## Use external data

```powershell
mosfet-platform update --project project.yaml --spec spec.yaml --database outputs/platform/catalog.sqlite3
mosfet-platform update --project project.yaml --spec spec.yaml --metrics metrics.csv --database outputs/platform/catalog.sqlite3
```

The project specifies the case manifest, measurement contract, base model,
and training options. Complete I-V data can train the model; calculated metric
tables are evaluated and stored without inventing training curves. Explicitly
predicted inputs are rejected as measured training evidence.

New models are trained in separate version directories. A failed training
attempt keeps valid measurements and the previous active model, records the
error, and exits with code 1. Independent validation is optional; NOT_RUN is
never reported as independent qualification. Use `--retrain` for an explicit
retry without changing the inputs.

See [the workflow and input contract](docs/workflow.md) for schemas, units,
validation policies, model versioning, and result definitions.

## Results and storage

`result.html` contains admission outcomes, training status, measured device
conclusions, and model comparisons. FALSE_PASS means the model predicts a pass
for a measured failure; the device conclusion remains FAIL. Invalid or missing
measurements cannot be replaced by predicted passes.

`catalog.sqlite3` stores measurement revisions, original curve bytes, hashes,
and run records. `models/` contains immutable published model versions and
training snapshots. These are operational assets; only the HTML is the user
entry point. Runtime files are ignored by Git.

For individual analysis tasks, the `evaluate`, `fit`, `predict`, and `compare`
commands remain available. Use the managed `update` workflow for repeated
admission and model publication.

## Verification

```powershell
python -m pytest -q
```

Tests cover curve/metric consistency, missing inputs, units, measured-reference
decisions, real fitting, duplicate admission, independent-data separation,
and failed retraining without losing the previous model. Publication checks
inspect version-controlled and unignored source files, not runtime outputs.

## Rights

Copyright (c) 2026 Dai Chenze. All rights reserved.

This repository is public solely for portfolio, demonstration, and evaluation.
No permission is granted to use, copy, modify, distribute, sublicense, or
create derivative works. See [LICENSE](LICENSE).
