# AI-Enabled Visual Waste Segregation

Classifies photos of waste into 6 categories (cardboard, food organics, glass, metal, miscellaneous
trash and paper) and wraps the model in a full MLOps pipeline: versioned data, tracked
experiments, a REST API, a demo UI, containers, Kubernetes manifests and CI/CD.

![CI/CD](https://github.com/Caffine-addict/AI-enabled-Visual-Waste-Segregation-/actions/workflows/ci-cd.yml/badge.svg)

## Results

Trained on the [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) dataset (3,077 images
across the 6 classes used here), with class-balanced training.

| Model | Input | Validation accuracy | AUC |
|---|---|---|---|
| **EfficientNetV2-S**, 3-stage fine-tuning, 50 epochs | 300×300 | **96.3%** | **0.997** |
| MobileNetV2 baseline | 224×224 | 83.9% | 0.94 |

| Class (EfficientNetV2-S) | Precision | Recall | F1 |
|---|---|---|---|
| Cardboard | 0.94 | 0.97 | 0.96 |
| Food Organics | 1.00 | 1.00 | 1.00 |
| Glass | 1.00 | 0.97 | 0.98 |
| Metal | 0.97 | 0.97 | 0.97 |
| Miscellaneous Trash | 0.96 | 0.91 | 0.93 |
| Paper | 0.92 | 0.97 | 0.95 |

Full comparison against ResNet50, VGG16, InceptionV3 and DenseNet121: [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md).

> **Which model the API serves:** by default `model.keras`, the MobileNetV2 baseline. The
> EfficientNetV2-S weights are in `artifacts/model/best_efficientnet.keras` and expect 300×300
> input.

## Pipeline

| Stage | Tooling |
|---|---|
| Data versioning | DVC (`data/RealWaste.dvc`, `mlruns.dvc`) |
| Training and tracking | TensorFlow / Keras, MLflow (`src/realwaste_mlops/training/`) |
| Evaluation | `src/realwaste_mlops/evaluation/evaluate.py` |
| Serving | FastAPI: `/health`, `/predict` (file upload), `/predict/base64`, `/classes` |
| Demo UI | Gradio, served from the API at `/gradio` |
| Containers | Docker, Docker Compose (API, MLflow, Jupyter; Nginx in the production compose file) |
| Orchestration | Kubernetes (`k8s/deployment.yaml`) |
| CI/CD | GitHub Actions: lint (Ruff), tests, DVC pull, image build and push |

## Run it

```bash
make install        # pip install -r requirements.txt
make api            # API at http://localhost:8000
make test           # pytest with coverage
make run            # full stack via Docker Compose (API on :8000, MLflow on :5000)
make train          # retrain (transfer learning), logged to MLflow
```

Classify an image:

```bash
curl -F "file=@bottle.jpg" http://localhost:8000/predict
```

The dataset is tracked with DVC. Set up your own DVC remote, or download RealWaste and place it in
`data/RealWaste/`, before training.

## Project layout

```
src/realwaste_mlops/
  api/          FastAPI app
  training/     training scripts (transfer, balanced, focal loss, EfficientNet)
  evaluation/   metrics and reports
  inference/    model loading and prediction
  features/     preprocessing
  ui/           Gradio demo
tests/          API, config, inference and preprocessing tests
k8s/  nginx/  configs/  artifacts/
```

## License

MIT
