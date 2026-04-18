# AI-Enabled Visual Waste Segregation

A production-ready MLOps project for automated waste classification using deep learning. This project classifies waste images into 6 categories: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, and Paper.

![Python](https://img.shields.io/badge/python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![MLflow](https://img.shields.io/badge/MLflow-2.20-purple)
![License](https://img.shields.io/badge/license-MIT-yellow)

## 🚀 Features

- **Deep Learning Model**: MobileNetV2 transfer learning with class-balanced training
- **High Accuracy**: 82%+ validation accuracy across all 6 waste categories
- **Production Ready**: Complete MLOps pipeline with Docker, Kubernetes, and CI/CD
- **API Server**: FastAPI-based REST API for real-time inference
- **Web Interface**: Gradio-based demo UI for easy testing
- **Experiment Tracking**: MLflow integration for monitoring training
- **Data Versioning**: DVC for dataset management
- **Containerized**: Docker and Docker Compose support

## 📊 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cardboard | 0.78 | 0.91 | 0.84 |
| Food Organics | 0.75 | 1.00 | 0.86 |
| Glass | 0.77 | 0.92 | 0.84 |
| Metal | 0.90 | 0.74 | 0.81 |
| Miscellaneous Trash | 0.82 | 0.68 | 0.74 |
| Paper | 0.91 | 0.79 | 0.84 |

**Overall Accuracy**: 82.17%

## 🏗️ Project Structure

```
AI-enabled-Visual-Waste-Segregation-/
├── src/                         # Source code
│   └── realwaste_mlops/
│       ├── api/                 # FastAPI application
│       ├── config/              # Configuration management
│       ├── data/                # Data processing
│       ├── features/            # Feature engineering
│       ├── inference/           # Model inference
│       ├── training/            # Training scripts
│       └── evaluation/          # Model evaluation
├── tests/                       # Unit tests
├── data/                        # Dataset (RealWaste)
├── configs/                     # Configuration files
├── k8s/                         # Kubernetes manifests
├── nginx/                       # Nginx configuration
├── artifacts/                   # Model artifacts
├── Dockerfile                   # Docker image definition
├── docker-compose.yml          # Development environment
├── docker-compose.prod.yml     # Production environment
├── Makefile                     # Development commands
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- TensorFlow 2.18+
- Git

### Setup

1. **Clone the repository**:
```bash
git clone git@github.com:Caffine-addict/AI-enabled-Visual-Waste-Segregation-.git
cd AI-enabled-Visual-Waste-Segregation-
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🎯 Quick Start

### Training the Model

```bash
# With MLflow tracking
make train-local-mlflow

# Simple training (50 epochs)
PYTHONPATH=src python -m realwaste_mlops.training.train_transfer
```

### Running the API

```bash
# Using Make
make run

# Or directly
PYTHONPATH=src python -m realwaste_mlops.api.main
```

### Access the Services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Gradio Demo | http://localhost:8000/gradio-demo |
| MLflow | http://localhost:5000 |

## 🔧 Usage

### Python API

```python
from realwaste_mlops.inference import predict

# Load classifier
classifier = predict.get_classifier()

# Predict
result = classifier.predict("path/to/image.jpg")
print(result["predicted_label"])  # e.g., "Glass"
print(result["confidence"])       # e.g., 0.98
```

### REST API

```bash
# Single prediction
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict

# Base64 prediction
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"base64_string","format":"base64"}' \
  http://localhost:8000/predict/base64

# Check health
curl http://localhost:8000/health

# List classes
curl http://localhost:8000/classes
```

## 🐳 Docker Deployment

### Development

```bash
docker-compose up
```

### Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Build Image

```bash
docker build -t realwaste-api:latest .
```

## ☸️ Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
```

## 🔄 CI/CD

The project includes GitHub Actions workflows for:
- Linting and testing
- Docker image building
- Model registration to MLflow
- Deployment to production

## 📈 Monitoring

### MLflow

Access MLflow tracking server at http://localhost:5000 to:
- View experiment runs
- Compare metrics
- Register models

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "service": "realwaste-classifier",
  "status": "ready",
  "model_loaded": true
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 📝 Configuration

Configuration files in `configs/`:
- `base.yaml` - Base configuration
- Environment-specific configs available

## 🎓 Dataset

This project uses the **RealWaste** dataset from Kaggle:
- **Source**: https://www.kaggle.com/datasets/aryashashank/realwaste
- **Classes**: 6 (Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper)
- **Total Images**: ~3,077

### Data Structure
```
data/
└── realwaste-main/
    └── RealWaste/
        ├── Cardboard/
        ├── Food Organics/
        ├── Glass/
        ├── Metal/
        ├── Miscellaneous Trash/
        └── Paper/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [RealWaste Dataset](https://www.kaggle.com/datasets/aryashashank/realwaste) for the dataset
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [MLflow](https://mlflow.org/) for experiment tracking
- [DVC](https://dvc.org/) for data version control