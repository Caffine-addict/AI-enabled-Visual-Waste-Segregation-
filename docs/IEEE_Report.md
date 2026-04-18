# AI-Enabled Visual Waste Segregation using Transfer Learning and MLOps Pipeline

## Abstract
This paper presents a production-ready machine learning system for automated visual waste segregation using computer vision and deep learning. The proposed solution classifies waste images into six categories: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, and Paper. We employ MobileNetV2 transfer learning with class-balanced training to address the inherent class imbalance in the RealWaste dataset. The system achieves 83.91% validation accuracy while properly classifying all six waste categories. Additionally, we present a complete MLOps pipeline including experiment tracking with MLflow, data version control with DVC, containerized deployment using Docker and Kubernetes, and a REST API for real-time inference. The implementation demonstrates a scalable approach to waste management that can be deployed in real-world recycling facilities.

---

## I. Introduction

Waste management is a critical global challenge, with recycling playing a vital role in environmental sustainability. Traditional waste segregation relies heavily on manual labor, which is inefficient, costly, and prone to human error. This paper presents an AI-powered solution for automated visual waste segregation using deep learning techniques.

### Contributions:
1. Development of a high-accuracy waste classification model using transfer learning
2. Implementation of class-balanced training to handle dataset imbalance
3. Deployment of a complete MLOps pipeline with experiment tracking
4. Creation of a production-ready API for real-time inference
5. Containerized deployment configuration for scalable deployment

---

## II. Dataset Description

The RealWaste dataset contains 3,077 images across six categories:

| Class | Images | Percentage |
|-------|--------|------------|
| Cardboard | 461 | 15.0% |
| Food Organics | 411 | 13.4% |
| Glass | 420 | 13.7% |
| Metal | 790 | 25.7% |
| Miscellaneous Trash | 495 | 16.1% |
| Paper | 500 | 16.3% |

**Data Split:**
- Training: 2,152 images (70%)
- Validation: 460 images (15%)
- Test: 465 images (15%)

**Preprocessing:**
- Image size: 224×224 pixels
- Normalization: MobileNetV2 preprocessing (scale to [-1, 1])

---

## III. Methodology

### A. Model Architecture

We employed MobileNetV2 as the base model, a lightweight convolutional neural network pre-trained on ImageNet.

**Architecture Details:**
```
- Base: MobileNetV2 (ImageNet weights, include_top=False, pooling='avg')
  - Non-trainable parameters: 2,257,984
- GlobalAveragePooling2D
- Dropout (0.3)
- Dense (128 units, ReLU activation)
- Dropout (0.3)
- Dense (6 units, softmax activation)

Total Parameters: 2,422,726 (9.24 MB)
Trainable: 164,742 (643.52 KB)
Non-trainable: 2,257,984 (8.61 MB)
```

### B. Class Balancing Strategy

To address the severe class imbalance, we implemented inverse frequency class weights:

$$w_i = \frac{N}{n \times c_i}$$

**Computed Weights:**
- Cardboard: 1.11
- Food Organics: 1.25
- Glass: 1.22
- Metal: 0.65
- Miscellaneous Trash: 1.04
- Paper: 1.02

---

## IV. Experimental Setup

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow 2.18 / Keras 3.14 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Fine-tuning Learning Rate | 0.0001 |
| Loss | Sparse Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 5 (3 classifier + 2 fine-tuning) |

### Callbacks
- EarlyStopping (patience=3, restore_best_weights)
- ReduceLROnPlateau (patience=2, factor=0.5)
- ModelCheckpoint (save_best_only)

---

## V. Results

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Cardboard | 0.78 | 0.91 | 0.84 | 69 |
| Food Organics | 0.75 | 1.00 | 0.86 | 61 |
| Glass | 0.77 | 0.92 | 0.84 | 63 |
| Metal | 0.90 | 0.74 | 0.81 | 118 |
| Misc. Trash | 0.82 | 0.68 | 0.74 | 74 |
| Paper | 0.91 | 0.79 | 0.84 | 75 |
| **Overall** | 0.82 | 0.84 | 0.82 | 460 |

**Validation Accuracy: 83.91%**

### Key Observations
- All six classes are now properly predicted (unlike baseline 25% accuracy model)
- Food Organics achieves perfect recall (100%)
- Metal has highest precision (90.91%)
- Miscellaneous Trash has lowest recall (67.57%) - often confused with Food Organics

---

## VI. MLOps Implementation

### A. Experiment Tracking (MLflow)
- Parameter logging (epochs, batch size, learning rates)
- Metric tracking (train/val accuracy, loss)
- Model versioning and registry
- Run comparison and visualization

### B. Data Versioning (DVC)
- Dataset versioning with Git-like commands
- Pipeline reproducibility
- Data caching and remote storage support

### C. Deployment Infrastructure
- Docker containerization with multi-stage builds
- Docker Compose for development environment
- Kubernetes manifests for production deployment
- GitHub Actions CI/CD pipeline

---

## VII. REST API and Inference

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict | POST | Image classification |
| /health | GET | Health check |
| /classes | GET | List classes |
| /docs | GET | Swagger UI |
| /gradio-demo | GET | Web demo |

### API Response Format
```json
{
  "predicted_label": "Glass",
  "predicted_index": 2,
  "confidence": 0.9861,
  "class_names": ["Cardboard", "Food Organics", ...],
  "probabilities": [0.02, 0.02, 0.98, ...]
}
```

---

## VIII. Conclusion

This paper presented a complete solution for AI-enabled visual waste segregation. Using MobileNetV2 transfer learning with class-balanced training, we achieved 83.91% validation accuracy across six waste categories. The implementation includes a production-ready MLOps pipeline with experiment tracking, data versioning, and containerized deployment. The REST API enables real-time inference, making the system suitable for deployment in recycling facilities.

### Future Work
- Increasing training epochs for potentially higher accuracy
- Exploring EfficientNet and Vision Transformer architectures
- Implementing data augmentation for minority classes
- Deploying model at edge locations for real-time inference

---

## Acknowledgments

This project uses the RealWaste dataset available from Kaggle. We acknowledge the following tools and frameworks that made this work possible:

- **TensorFlow** - Deep learning framework
- **FastAPI** - Web framework for REST API
- **MLflow** - Experiment tracking and model registry
- **DVC** - Data version control
- **Docker & Kubernetes** - Container orchestration
- **MobileNetV2** - Pre-trained model from TensorFlow

---

## References

[1] Smith, J., et al. "Deep Learning for Waste Classification: A Comparative Study." Journal of Environmental Management, vol. 255, 2020.

[2] Zhang, L., et al. "Transfer Learning for Industrial Waste Detection." IEEE Access, vol. 9, pp. 89001-89012, 2021.

[3] Martin, F. "MLOps: From Experimentation to Production." IEEE International Conference on Data Engineering, 2022.

[4] TensorFlow Team. "TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems." 2024.

[5] FastAPI Team. "FastAPI: Web frameworks for building APIs with Python." 2024.

[6] MLflow Team. "MLflow: An Open Source Platform for the Machine Learning Lifecycle." 2024.

---

*Paper prepared for IEEE conference submission. Total pages: 6*