# RealWaste Model Contract

## Purpose
Define the shared contract used by training, evaluation, FastAPI, and Gradio so all components load and serve the same model consistently.

## Input Contract

### Raw input
- Type: image file or PIL image
- Accepted sources: uploaded image, webcam capture, API multipart upload
- Expected color mode: RGB

### Preprocessed tensor
- Shape: `(224, 224, 3)`
- Dtype: `float32`
- Resize policy: direct resize to `224x224`
- Preprocessing owner: `realwaste_mlops.features.preprocess`
- Rule: training, evaluation, and inference must call the same preprocessing code path

## Output Contract

### Prediction response
- `predicted_label`: `str`
- `predicted_index`: `int`
- `confidence`: `float`
- `probabilities`: `list[float]`
- `class_names`: `list[str]`

### FastAPI response shape
```json
{
  "predicted_label": "Glass",
  "predicted_index": 2,
  "confidence": 0.97,
  "probabilities": [0.01, 0.01, 0.97, 0.00, 0.01, 0.00],
  "class_names": [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper"
  ]
}
```

## Artifact Contract

### Primary model artifact
- Path: `artifacts/model/model.keras`

### Model metadata
- Path: `artifacts/model/model_metadata.json`
- Required fields:
  - `model_name`
  - `model_path`
  - `image_size`
  - `class_names`
  - `preprocessing`
  - `metrics`
  - `mlflow_run_id`
  - `promoted`

### Example metadata
```json
{
  "model_name": "realwaste-efficientnetv2b2",
  "model_path": "artifacts/model/model.keras",
  "image_size": [224, 224],
  "class_names": [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper"
  ],
  "preprocessing": {
    "color_mode": "RGB",
    "resize": [224, 224],
    "dtype": "float32"
  },
  "metrics": {
    "accuracy": 0.0,
    "macro_f1": 0.0
  },
  "mlflow_run_id": "",
  "promoted": false
}
```

## Evaluation Contract
- Evaluation reads the promoted model metadata.
- Metrics are computed from held-out test manifests.
- Summary path: `reports/evaluation_summary.json`
- Promotion decision path: `artifacts/model/model_metadata.json`

## Serving Contract

### Health endpoint
- Path: `/health`
- Returns service and model readiness.

### Predict endpoint
- Path: `/predict`
- Request: multipart image upload
- Response: JSON prediction payload using the output contract above.

### Gradio
- Mounted at `/gradio`
- Must call the same inference service used by FastAPI.

## Test Expectations
- Config tests verify contract paths and defaults.
- Data tests verify split determinism and validation failures.
- Training/evaluation tests verify metadata shape and promotion rules.
- API tests verify health, validation, and prediction response schema.
- UI tests smoke-test Gradio app creation without manual browser interaction.
