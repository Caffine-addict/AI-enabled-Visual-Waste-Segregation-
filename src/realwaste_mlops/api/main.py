"""FastAPI server for RealWaste classifier with Gradio UI."""

import io
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from fastapi.responses import Response
from PIL import Image

from realwaste_mlops.inference import predict
from realwaste_mlops.features import preprocess


app = FastAPI(
    title="RealWaste Classifier API",
    description="EfficientNetV2B2-based waste classification API with Gradio UI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.classifier = None
app.state.gradio_app = None


@app.middleware("http")
async def load_classifier_on_request(request, call_next):
    """Lazy load classifier on first request."""
    if app.state.classifier is None:
        try:
            app.state.classifier = predict.get_classifier()
        except FileNotFoundError:
            pass
    return await call_next(request)


@app.get("/health")
async def health():
    """Health check endpoint."""
    ready = app.state.classifier is not None
    return {
        "service": "realwaste-classifier",
        "status": "ready" if ready else "model_not_loaded",
        "model_loaded": ready,
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict waste class from uploaded image.

    Request: multipart image upload
    Response: JSON prediction payload
    """
    if app.state.classifier is None:
        return JSONResponse(
            {"error": "Model not loaded. Train a model first."},
            status_code=503,
        )

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    try:
        result = app.state.classifier.predict(image)
        return result
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


@app.post("/predict/base64")
async def predict_base64(request: dict):
    """Predict waste class from base64 encoded image.

    Request body: {"image": "<base64_string>", "format": "base64"}
    """
    if app.state.classifier is None:
        return JSONResponse(
            {"error": "Model not loaded. Train a model first."},
            status_code=503,
        )

    import base64

    image_data = request.get("image", "")
    image_format = request.get("format", "base64")

    if image_format != "base64":
        return JSONResponse(
            {"error": "Only base64 format supported"},
            status_code=400,
        )

    try:
        # Decode base64
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        result = app.state.classifier.predict(image)
        return result
    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500,
        )


@app.get("/classes")
async def list_classes():
    """List available classes."""
    if app.state.classifier is None:
        return {"class_names": preprocess.CLASS_NAMES}
    return {"class_names": app.state.classifier.class_names}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RealWaste Classifier API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST multipart form)",
            "predict_base64": "/predict/base64 (POST JSON)",
            "classes": "/classes (GET)",
            "docs": "/docs",
            "gradio": "/gradio",
        },
    }


def create_gradio_interface() -> "gradio.Interface":
    """Create Gradio interface for the classifier."""
    import gradio as gr

    def predict_fn(image):
        if image is None:
            return {"error": "No image provided"}

        if app.state.classifier is None:
            return {"error": "Model not loaded. Train a model first."}

        try:
            result = app.state.classifier.predict(image)
            # Format output nicely
            return {
                "predicted_class": result["predicted_label"],
                "confidence": f"{result['confidence']:.2%}",
                "all_probabilities": dict(
                    zip(
                        result["class_names"],
                        [f"{p:.2%}" for p in result["probabilities"]],
                    )
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    # Build class choices for examples
    class_choices = preprocess.CLASS_NAMES

    interface = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil", label="Waste Image"),
        outputs=gr.JSON(label="Prediction Result"),
        title="RealWaste Classifier",
        description="Upload an image of waste to classify it into one of 6 categories: "
        "Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, or Paper.",
        examples=None,  # Could add example images
        cache_examples=False,
    )

    return interface


@app.on_event("startup")
async def startup_event():
    """Initialize Gradio app on startup."""
    try:
        gradio_interface = create_gradio_interface()
        app.state.gradio_app = gradio_interface
        print("Gradio interface initialized")
    except Exception as e:
        print(f"Could not initialize Gradio: {e}")
        app.state.gradio_app = None


# Create a simple wrapper to mount Gradio
# We'll use a separate route handler instead of mounting
@app.get("/gradio", response_class=HTMLResponse)
async def gradio_ui():
    """Serve Gradio UI."""
    if app.state.gradio_app is None:
        return HTMLResponse(
            content="<html><body><h1>Gradio UI not available</h1>"
            "<p>Model may not be loaded. Train a model first.</p></body></html>",
            status_code=503,
        )

    # Get the Gradio HTML
    try:
        from gradio.routes import create_app

        # For Gradio 5.x, we need to create a separate gradio app and mount it
        # Since this is complex, we'll provide a redirect to standalone gradio
        return HTMLResponse(
            content="""
            <html>
            <head>
                <title>RealWaste Classifier - Gradio UI</title>
                <meta http-equiv="refresh" content="0;url=/gradio-demo" />
            </head>
            <body>
                <h1>Redirecting to Gradio Interface...</h1>
                <p>If not redirected, <a href="/gradio-demo">click here</a></p>
            </body>
            </html>
            """,
            media_type="text/html",
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>",
            status_code=500,
        )


@app.get("/gradio-demo", response_class=HTMLResponse)
async def gradio_demo():
    """Serve Gradio demo (standalone page)."""
    # Create inline Gradio for direct embedding
    import gradio as gr

    def predict_fn(image):
        if image is None:
            return {"error": "No image provided"}

        if app.state.classifier is None:
            return {"error": "Model not loaded. Train a model first."}

        try:
            result = app.state.classifier.predict(image)
            return result
        except Exception as e:
            return {"error": str(e)}

    # Build a simple HTML page with embedded Gradio
    # For simplicity, we'll serve the Gradio app directly as a sub-app

    # Actually, let's create a simple workaround - serve gradio blocks as HTML
    # This is a simpler approach that doesn't require complex mounting

    # Get list of available sample images if model is loaded
    sample_images = ""
    if app.state.classifier:
        try:
            manifest_path = (
                Path(__file__).parent.parent.parent.parent
                / "artifacts"
                / "manifests"
                / "train.json"
            )
            if manifest_path.exists():
                import json

                with open(manifest_path) as f:
                    data = json.load(f)
                    # Get one image per class for examples
                    seen_classes = set()
                    examples = []
                    for record in data["records"]:
                        if record["class_name"] not in seen_classes:
                            seen_classes.add(record["class_name"])
                            examples.append(record["file_path"])
                            if len(examples) >= 6:
                                break

                    if examples:
                        sample_images = ",".join([f'"{e}"' for e in examples])
        except Exception:
            pass

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RealWaste Classifier</title>
        <script src="https://cdn.jsdelivr.net/npm/gradio@5.0.0"></script>
        <style>
            body {{ font-family: system-ui, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; }}
            .status {{ padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .status.ready {{ background: #d4edda; color: #155724; }}
            .status.not-ready {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RealWaste Classifier</h1>
            <div class="status {"ready" if app.state.classifier else "not-ready"}">
                Model: {"Loaded" if app.state.classifier else "Not Loaded"}
            </div>
            <div id="gradio-app"></div>
        </div>
        <script>
            const app = gradioApp({{
                target: document.getElementById('gradio-app'),
                title: "RealWaste Classifier",
                examples: [{sample_images}],
            }});
            
            // Define the prediction function
            async function predict(image) {{
                if (!image) return {{error: "No image provided"}};
                if (!{app.state.classifier is not None}) return {{error: "Model not loaded"}};
                
                const formData = new FormData();
                formData.append('file', image);
                
                const response = await fetch('/predict', {{ method: 'POST', body: formData }});
                return await response.json();
            }}
        </script>
    </body>
    </html>
    """

    # Since direct Gradio embedding is complex, let's use a simpler approach
    # Serve a simple upload form that calls our API
    simple_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RealWaste Classifier - Upload Demo</title>
        <style>
            body { font-family: system-ui, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .upload-section { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .result { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 8px; }
            .result-item { margin: 10px 0; }
            .confidence { font-weight: bold; color: #27ae60; }
            .error { color: #e74c3c; }
            .probs { margin-top: 15px; }
            .prob-bar { display: flex; align-items: center; margin: 5px 0; }
            .prob-label { width: 150px; }
            .prob-value { width: 80px; text-align: right; }
            .prob-track { flex: 1; height: 20px; background: #eee; border-radius: 4px; overflow: hidden; }
            .prob-fill { height: 100%; background: #3498db; }
        </style>
    </head>
    <body>
        <h1>RealWaste Classifier</h1>
        <p>Upload an image to classify waste into 6 categories.</p>
        
        <div class="upload-section">
            <input type="file" id="file-input" accept="image/*">
            <br><br>
            <button onclick="predict()">Classify</button>
        </div>
        
        <div id="result" style="display: none;"></div>
        
        <script>
            async function predict() {
                const fileInput = document.getElementById('file-input');
                const resultDiv = document.getElementById('result');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<p>Processing...</p>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                        return;
                    }
                    
                    let html = '<div class="result">';
                    html += '<h2>Prediction: ' + data.predicted_label + '</h2>';
                    html += '<div class="result-item">Confidence: <span class="confidence">' + 
                            (data.confidence * 100).toFixed(2) + '%</span></div>';
                    
                    html += '<div class="probs"><h3>All Probabilities:</h3>';
                    data.probabilities.forEach((p, i) => {
                        const width = (p * 100).toFixed(1);
                        html += '<div class="prob-bar">';
                        html += '<span class="prob-label">' + data.class_names[i] + '</span>';
                        html += '<span class="prob-value">' + (p * 100).toFixed(1) + '%</span>';
                        html += '<div class="prob-track"><div class="prob-fill" style="width: ' + width + '%"></div></div>';
                        html += '</div>';
                    });
                    html += '</div>';
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                } catch (e) {
                    resultDiv.innerHTML = '<div class="error">Error: ' + e + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=simple_html, media_type="text/html")


def run(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
