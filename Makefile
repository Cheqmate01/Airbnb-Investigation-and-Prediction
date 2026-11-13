PY := .venv/bin/python

.PHONY: setup build audit eda analyse model report notebooks all test clean

setup:
	python3.9 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m ipykernel install --user --name airbnb-week1 --display-name "Airbnb Week 1"

build:
	$(PY) -m src.build_dataset

audit:
	$(PY) -m src.audit

eda:
	$(PY) -m src.eda

analyse:
	$(PY) -m src.analysis

model:
	$(PY) -m src.modelling
	$(PY) -m src.reporting

report:
	$(PY) -m src.reporting

notebooks:
	$(PY) scripts/make_notebooks.py --execute

all: build audit eda analyse model notebooks

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim/* data/processed/* outputs/figures/* outputs/tables/* outputs/models/*
