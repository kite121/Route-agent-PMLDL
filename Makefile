PYTHON := .venv/bin/python
COMPOSE := docker compose -f code/deployment/docker-compose.yml
AIRFLOW_HOME := $(CURDIR)/services/airflow/runtime
AIRFLOW_DAGS := $(CURDIR)/services/airflow/dags

.PHONY: install data train api app compose-up compose-down airflow airflow-test smoke

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) code/datasets/download_data.py
	$(PYTHON) code/datasets/create_batches.py
	$(PYTHON) code/datasets/validate_data.py
	$(PYTHON) code/datasets/prepare_data.py

train:
	$(PYTHON) code/models/build_training_pairs.py
	$(PYTHON) code/models/train.py
	$(PYTHON) code/models/evaluate.py
	$(PYTHON) code/models/quality_gate.py
	$(PYTHON) code/models/package_model.py

api:
	PYTHONPATH=code $(PYTHON) -m uvicorn deployment.api.main:app --host 127.0.0.1 --port 8000

app:
	PYTHONPATH=code/deployment API_BASE_URL=http://127.0.0.1:8000 $(PYTHON) -m streamlit run code/deployment/app/app.py

compose-up:
	$(COMPOSE) up --build -d

compose-down:
	$(COMPOSE) down

airflow:
	AIRFLOW_HOME=$(AIRFLOW_HOME) AIRFLOW__CORE__DAGS_FOLDER=$(AIRFLOW_DAGS) $(PYTHON) -m airflow standalone

airflow-test:
	$(PYTHON) -m py_compile services/airflow/dags/agent_router_pipeline.py

smoke:
	$(PYTHON) code/deployment/smoke_check.py
