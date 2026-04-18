"""Gradio UI for RealWaste classifier."""

from __future__ import annotations

from typing import Optional

import gradio as gr
from PIL import Image

from realwaste_mlops.inference import predict
from realwaste_mlops.features import preprocess


def create_gradio_interface(classifier: Optional[predict.Classifier] = None):
    """Create Gradio interface for the classifier.

    Args:
        classifier: Classifier instance (optional, will create if not provided)

    Returns:
        Gradio Blocks interface
    """
    if classifier is None:
        try:
            classifier = predict.get_classifier()
        except FileNotFoundError:
            classifier = None

    class_names = classifier.class_names if classifier else preprocess.CLASS_NAMES

    def classify(image: Image.Image):
        """Classify an image."""
        if classifier is None:
            return "Model not loaded. Train a model first.", None, None

        result = classifier.predict(image, return_probs=True)

        return (
            result["predicted_label"],
            {c: p for c, p in zip(class_names, result["probabilities"])},
            result["confidence"],
        )

    with gr.Blocks(title="RealWaste Classifier") as interface:
        gr.Markdown("# RealWaste Image Classifier")
        gr.Markdown("Upload an image to classify waste type.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            with gr.Column():
                label_output = gr.Label(label="Predicted Class")
                probs_output = gr.Label(label="Probabilities", type="confidence")
                conf_output = gr.Number(label="Confidence")

        image_input.change(
            classify,
            inputs=image_input,
            outputs=[label_output, probs_output, conf_output],
        )

        gr.Examples(
            examples=[],
            inputs=image_input,
        )

    return interface


def get_interface() -> gr.Blocks:
    """Get the Gradio interface (convenience function)."""
    return create_gradio_interface()
