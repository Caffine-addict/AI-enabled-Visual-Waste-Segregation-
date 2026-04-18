# RealWaste MLOps - Deployment Makefile
# Usage: make <target>

.PHONY: help install test lint build run stop clean deploy

help:
	@echo "RealWaste MLOps - Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint      - Lint code"
	@echo "  make build     - Build Docker image"
	@echo "  make run       - Run locally with docker-compose"
	@echo "  make stop      - Stop containers"
	@echo "  make clean     - Clean up containers and volumes"
	@echo "  make deploy    - Deploy to Kubernetes"
	@echo "  make logs      - View API logs"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/ tests/

build:
	docker build -t realwaste-api:latest .

run:
	docker-compose up -d
	@echo "API available at: http://localhost:8000"
	@echo "MLflow available at: http://localhost:5000"
	@echo "Jupyter available at: http://localhost:8888"

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

logs:
	docker-compose logs -f api

deploy-k8s:
	kubectl apply -f k8s/deployment.yaml

undeploy-k8s:
	kubectl delete -f k8s/deployment.yaml

# Training commands
train:
	PYTHONPATH=src python -m realwaste_mlops.training.train_transfer

train-local-mlflow:
	PYTHONPATH=src MLFLOW_TRACKING_URI=http://localhost:5000 python -m realwaste_mlops.training.train_transfer

# API commands
api:
	PYTHONPATH=src python -m realwaste_mlops.api.main

# Evaluation
evaluate:
	PYTHONPATH=src python -m realwaste_mlops.evaluation.evaluate