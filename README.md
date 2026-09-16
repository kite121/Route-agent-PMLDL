# Route-agent-PMLDL

Automated MLOps pipeline for a multilingual semantic tool router. The project
fine-tunes `intfloat/multilingual-e5-small` to map a user request either to a
registered agent tool or to the fallback `no_tool` decision.

## Architecture

```text
Amazon MASSIVE 1.1 batch
        ↓
validation and preparation
        ↓
training pairs → fine-tuning → evaluation → quality gate
        ↓
agent_router.tar.gz + model_manifest.json
        ↓
FastAPI (:8000) ← HTTP → Streamlit (:8501)
```

Airflow schedules the same chain and processes no more than one immutable
training batch per run. ClearML stores every training, evaluation, quality-gate
and packaging experiment, including rejected candidates.

## Data and batching

The source is the pinned `AmazonScience/massive`, configuration `all_1.1`,
revision `e9957e4e8eb17fc67e3faf1a908829895653e775`. The test and validation
splits are fixed. The train split is deterministically shuffled into batches of
200 examples; each prepared dataset version records its source revision,
included batch IDs, row counts and SHA256 hashes in `data/manifests/`.

The committed demo configuration intentionally keeps each run short: one
training epoch per batch. Evaluation uses reproducible uniform samples of
2,000 examples from validation and 2,000 from test; the splits remain separate
and no evaluation sample is used for training.

## Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
make install
cp .env.example .env
```

Add the ClearML access key and secret to `.env`. Do not commit this file.
The remaining ClearML host values in `.env.example` are correct for the hosted
ClearML service.

| Variable | Purpose |
| --- | --- |
| `CLEARML_API_ACCESS_KEY`, `CLEARML_API_SECRET_KEY` | Authentication for experiment and artifact tracking. |
| `CLEARML_API_HOST`, `CLEARML_WEB_HOST`, `CLEARML_FILES_HOST` | ClearML hosted-service endpoints. |
| `API_BASE_URL` | Streamlit endpoint for FastAPI; Compose sets `http://api:8000`. |

## Local pipeline

Process the next pending batch:

```bash
make data
```

Fine-tune, evaluate, apply the quality gate and package an accepted model:

```bash
make train
```

The accepted package is written locally to `models/current/` as
`agent_router.tar.gz` and `model_manifest.json`. Generated data, models, logs
and secrets are intentionally ignored by Git.

## Local API and UI

Start these in separate terminals:

```bash
make api
make app
```

- FastAPI: `http://127.0.0.1:8000`
- OpenAPI: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`

Streamlit never loads the encoder. It calls FastAPI through `API_BASE_URL`.

## Docker deployment

Docker Compose creates two containers: `api` contains FastAPI and inference
dependencies; `app` contains only Streamlit and its HTTP client. On the Compose
network, the UI calls `http://api:8000`. The model package is mounted read-only
into the API, while extraction uses an internal writable cache.

```bash
make compose-up
make smoke
```

The smoke check confirms `/health`, `/model-info`, a route to `alarm_set`, a
`no_tool` fallback, and that the served model version matches the current
manifest. Stop services with `make compose-down`.

## Airflow

Start Airflow locally:

```bash
make airflow
```

The DAG is `agent_router_pipeline`, runs every five minutes, has
`catchup=False` and `max_active_runs=1`. Its tasks are grouped into Data
Engineering, Model Engineering and Deployment. It passes only paths and
metadata between tasks; datasets and model artifacts remain on disk rather
than being placed in XCom. If no batch is pending, the run is skipped. If the
quality gate rejects a candidate, packaging and deployment do not run.

Airflow UI is available at `http://127.0.0.1:8080` while `make airflow` is
running. ClearML experiments are available at `https://app.clear.ml`.

Check DAG syntax without starting a run:

```bash
make airflow-test
```

## Quality and reproducibility

The model uses `OnlineContrastiveLoss` with balanced positive and negative
pairs. The fallback threshold is selected on validation Macro F1 and applied
unchanged to the held-out test sample. The quality gate requires test
`Accuracy@1 >= 0.70` and Macro F1 no lower than the base encoder. Each model
manifest includes dataset version and hashes, tool-registry hash, Git commit,
ClearML task IDs and evaluation metrics.

## Demonstration

1. Run `make compose-up` and `make smoke`.
2. Open Streamlit and route `Set an alarm for seven tomorrow morning`.
3. Show the selected `alarm_set` tool, alternatives, similarity score and
   model version.
4. Route `Tell me a funny joke about cats` and show the `no_tool` fallback.
5. In ClearML, show the linked training, evaluation, quality-gate and package
   tasks.
6. In Airflow, show the successful scheduled `agent_router_pipeline` run and
   its three task groups.
