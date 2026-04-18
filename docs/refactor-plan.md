# RealWaste Notebook Refactor Plan

## Source Audit

### Current inputs
- Notebook: `/Users/pritamwani/Downloads/minor_projectt.ipynb`
- Dataset archive: `/Users/pritamwani/Downloads/realwaste.zip`

### Current notebook flow
1. Installs dependencies inside the notebook.
2. Extracts dataset manually into Colab-specific `/content/...` paths.
3. Builds TensorFlow datasets directly from folder structure.
4. Trains an EfficientNetV2B2-based classifier.
5. Evaluates with confusion matrix and classification report.
6. Fine-tunes a second model variant and saves local `.keras` artifacts.
7. Runs a Gradio demo directly from notebook state.

### Notebook-only assumptions to remove
- Hardcoded Colab paths under `/content`
- Ad hoc package installation in cells
- Training, evaluation, and serving coupled to notebook execution order
- Manually duplicated class labels in deployment cells
- Local artifact names assumed to exist without metadata or registry

## Target Local Project

### Goals
- Reproducible local Python project
- ZIP-based dataset ingestion
- Deterministic train/val/test manifests
- Shared preprocessing and label contract across train/eval/inference
- MLflow experiment tracking
- DVC pipeline reproducibility
- FastAPI inference API with mounted Gradio UI
- Automated tests for core behavior and integrations

### Proposed layout
```text
realwaste-mlops/
├── configs/
│   └── base.yaml
├── data/
│   ├── raw/
│   │   └── realwaste.zip
│   ├── external/
│   └── processed/
├── artifacts/
│   ├── manifests/
│   └── model/
├── reports/
├── src/realwaste_mlops/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── inference/
│   ├── training/
│   └── ui/
├── tests/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
└── README.md
```

### Runtime entrypoints
- Data prep: `python -m realwaste_mlops.data.ingest`
- Training: `python -m realwaste_mlops.training.train`
- Evaluation: `python -m realwaste_mlops.evaluation.evaluate`
- API: `uvicorn realwaste_mlops.api.app:app --reload`
- DVC pipeline: `dvc repro`

## Tooling decisions

### MLflow
- Track params, metrics, plots, model artifact, and evaluation summary.
- Use local tracking by default via config.

### DVC
- Track pipeline stages for data prep, training, and evaluation.
- Centralize reproducible parameters in `params.yaml`.

### FastAPI
- Expose `/health` and `/predict` endpoints.
- Keep routes thin and delegate to inference service.

### Gradio
- Build a lightweight image upload/webcam demo.
- Mount into FastAPI at `/gradio`.

## Key risks and mitigations

### Preprocessing drift
- Risk: training and inference transform images differently.
- Mitigation: a single shared preprocessing module and model metadata contract.

### Label mismatch
- Risk: hardcoded labels diverge from dataset-derived labels.
- Mitigation: persist `class_names` in model metadata and load them everywhere.

### Artifact sprawl
- Risk: MLflow, DVC, and serving paths diverge.
- Mitigation: centralize artifact directories in config and metadata.

### Heavy tests
- Risk: tests become slow or require GPU.
- Mitigation: use small fixtures, mocks, and synthetic images for unit/integration tests.

## Delivery order
1. Define shared project and model contract.
2. Scaffold project and config.
3. Implement ZIP ingestion, validation, and split manifests.
4. Add config/data tests.
5. Implement training with MLflow.
6. Implement evaluation and promotion metadata.
7. Add training/evaluation tests.
8. Add DVC pipeline wiring.
9. Implement FastAPI inference service.
10. Mount Gradio UI.
11. Add API/inference/UI tests.
