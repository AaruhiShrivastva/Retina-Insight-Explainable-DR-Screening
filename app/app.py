"""RetinaInsight - Transfer Learning for Explainable Diabetic Retinopathy Screening.

Interactive Streamlit research screening-support application.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure local Keras cache is set before TensorFlow loads
_LOCAL_KERAS_CACHE = PROJECT_ROOT / ".keras"
if _LOCAL_KERAS_CACHE.is_dir() and "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = str(_LOCAL_KERAS_CACHE)

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

from configs.training_config import config
from src.database.db_manager import DatabaseManager
from src.explainability.gradcam import explain_prediction
from src.models.transfer_models import build_model

# Page configuration
st.set_page_config(
    page_title="RetinaInsight | DR Screening Support",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color and severity mappings
SEVERITY_COLORS = {
    0: "#28a745",  # Green - No DR
    1: "#17a2b8",  # Cyan - Mild
    2: "#ffc107",  # Yellow - Moderate
    3: "#fd7e14",  # Orange - Severe
    4: "#dc3545",  # Red - Proliferative DR
}

CLINICAL_RECOMMENDATIONS = {
    0: "Routine annual screening recommended. No active retinopathy signs detected.",
    1: "Re-screen within 6–12 months. Mild microaneurysms may be present; blood glucose control advised.",
    2: "Referral to an ophthalmologist recommended within 3–6 months for complete dilated exam.",
    3: "Urgent referral to a retinal specialist recommended within 2–4 weeks. High risk of progression.",
    4: "Immediate referral to a vitreoretinal specialist. Neovascularization poses imminent risk of vision loss.",
}


@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Cached singleton database manager."""
    return DatabaseManager()


@st.cache_resource
def load_screening_model(architecture: str) -> tf.keras.Model:
    """Load trained model weights from disk, or instantiate transfer model if final weights not found."""
    final_path = config.SAVED_MODELS_DIR / f"{architecture}_final.keras"
    if final_path.is_file():
        return tf.keras.models.load_model(final_path)
    # Fallback to checkpoint if available
    stage2_ckpt = config.CHECKPOINTS_DIR / f"{architecture}_stage2_best.keras"
    if stage2_ckpt.is_file():
        return tf.keras.models.load_model(stage2_ckpt)
    stage1_ckpt = config.CHECKPOINTS_DIR / f"{architecture}_stage1_best.keras"
    if stage1_ckpt.is_file():
        return tf.keras.models.load_model(stage1_ckpt)
    # Otherwise instantiate fresh pretrained model
    return build_model(architecture=architecture, trainable_backbone=False)


def render_header() -> None:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 2.2rem;">👁️ RetinaInsight</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Transfer Learning for Explainable Diabetic Retinopathy Screening Support (SIH26038)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, float]:
    st.sidebar.header("⚙️ Configuration")
    architecture = st.sidebar.selectbox(
        "Model Architecture",
        options=list(config.SUPPORTED_ARCHITECTURES),
        index=2,  # Default to MobileNetV2
        format_func=lambda x: {
            "mobilenetv2": "MobileNetV2 (Fast, Lightweight)",
            "efficientnetb0": "EfficientNetB0 (Compound Scaling)",
            "resnet50": "ResNet50 (Deep Residual Network)",
        }.get(x, x),
    )

    cam_alpha = st.sidebar.slider(
        "Grad-CAM Heatmap Intensity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Adjust the transparency of the saliency overlay on the fundus photograph.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        ### ℹ️ Research Notice
        This tool is an **exploratory screening-support application** developed for academic research.
        It is **not** an FDA-cleared diagnostic medical device. All outputs require clinical adjudication by a licensed ophthalmologist.
        """
    )
    return architecture, cam_alpha


def run_screening_tab(architecture: str, cam_alpha: float) -> None:
    st.subheader("🔬 Fundus Image Screening & Explainable AI")

    col_input, col_action = st.columns([2, 1])

    with col_input:
        input_mode = st.radio(
            "Select Image Source:",
            options=["Upload Image File", "Select Labeled Test Sample"],
            horizontal=True,
        )

        selected_image_path: Path | None = None
        uploaded_file = None

        if input_mode == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload a retinal fundus photograph (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                temp_upload_dir = config.PREDICTIONS_DIR / "uploads"
                temp_upload_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_upload_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_image_path = temp_path
        else:
            # Load samples from test metadata
            test_meta_path = config.TEST_METADATA_PATH
            if test_meta_path.is_file():
                test_df = pd.read_csv(test_meta_path)
                sample_options = {}
                for _, row in test_df.head(20).iterrows():
                    label = f"ID: {row['image_id']} — Actual: Grade {row['diagnosis']} ({row['diagnosis_name']})"
                    sample_options[label] = Path(row["image_path"])

                selected_label = st.selectbox("Choose a validated test image:", options=list(sample_options.keys()))
                if selected_label:
                    selected_image_path = sample_options[selected_label]

    if selected_image_path is None or not selected_image_path.is_file():
        st.info("👈 Please upload a fundus photograph or choose a sample to begin screening.")
        return

    # Load model and run inference
    with st.spinner(f"Running {architecture.upper()} inference and computing Grad-CAM saliency..."):
        model = load_screening_model(architecture)
        start_time = time.perf_counter()
        explanation = explain_prediction(
            model=model,
            image_path=selected_image_path,
            architecture=architecture,
            alpha=cam_alpha,
        )
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 1)

    # Save overlay image
    overlay_filename = f"pred_{Path(selected_image_path).stem}_{int(time.time())}.png"
    overlay_path = config.PREDICTIONS_DIR / overlay_filename
    Image.fromarray(explanation["overlay_image"]).save(overlay_path)

    # Log to SQLite
    db = get_db_manager()
    db.log_prediction(
        image_name=Path(selected_image_path).name,
        image_path=str(selected_image_path),
        model_name=architecture,
        predicted_grade=explanation["predicted_grade"],
        predicted_severity=explanation["predicted_name"],
        confidence=explanation["confidence"],
        probabilities=explanation["probabilities"],
        overlay_path=str(overlay_path),
        execution_time_ms=duration_ms,
    )

    # Display Results
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### 📷 Fundus Photograph Input")
        st.image(explanation["original_resized"], caption="Original Image (224×224 RGB)", use_container_width=True)

    with res_col2:
        st.markdown("#### 🔍 Grad-CAM Visual Explanation")
        st.image(
            explanation["overlay_image"],
            caption=f"Grad-CAM Heatmap ({architecture.upper()} Saliency)",
            use_container_width=True,
        )

    # Diagnostic Summary Box
    grade = explanation["predicted_grade"]
    severity_name = explanation["predicted_name"]
    color = SEVERITY_COLORS.get(grade, "#333333")

    st.markdown(
        f"""
        <div style="background-color: {color}20; border-left: 6px solid {color}; padding: 1.2rem; border-radius: 6px; margin: 1.5rem 0;">
            <h3 style="margin: 0; color: {color};">Screening Result: Grade {grade} — {severity_name}</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
                <b>Confidence:</b> {explanation['confidence'] * 100:.2f}% | <b>Inference Latency:</b> {duration_ms} ms
            </p>
            <p style="margin: 0.5rem 0 0 0; font-style: italic;">
                <b>Clinical Protocol Suggestion:</b> {CLINICAL_RECOMMENDATIONS.get(grade, '')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Class probability bar chart
    st.markdown("#### 📊 Predicted Class Probability Distribution")
    probs_df = pd.DataFrame({
        "Grade": [f"Grade {i}: {config.CLASS_NAMES[i]}" for i in range(5)],
        "Probability": explanation["probabilities"],
    })

    chart = (
        alt.Chart(probs_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Grade:N", sort=None, title="Diabetic Retinopathy Severity Grade"),
            y=alt.Y("Probability:Q", title="Predicted Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Grade:N", legend=None),
            tooltip=["Grade", alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def run_history_tab() -> None:
    st.subheader("📋 Screening Inference History")
    db = get_db_manager()
    df = db.get_recent_predictions(limit=100)

    if df.empty:
        st.info("No predictions recorded yet. Run a screening to log inference history.")
        return

    st.dataframe(
        df[["prediction_uid", "timestamp", "image_name", "model_name", "predicted_grade", "predicted_severity", "confidence", "execution_time_ms"]],
        use_container_width=True,
    )

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download History as CSV",
        data=csv_data,
        file_name="retinainsight_prediction_history.csv",
        mime="text/csv",
    )


def run_analytics_tab() -> None:
    st.subheader("📈 Screening Analytics & Model Usage Dashboard")
    db = get_db_manager()
    summary = db.get_analytics_summary()

    total = summary["total_predictions"]
    if total == 0:
        st.info("No screening analytics recorded yet. Perform image screenings to populate the dashboard.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Screenings Completed", total)
    m2.metric("Mean Prediction Confidence", f"{summary['avg_confidence'] * 100:.1f}%")

    high_risk_count = summary["grade_counts"].get("Severe", 0) + summary["grade_counts"].get("Proliferative DR", 0)
    high_risk_pct = (high_risk_count / total) * 100 if total > 0 else 0
    m3.metric("High-Risk Cases (Grade 3/4)", f"{high_risk_pct:.1f}%")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### Diagnosis Severity Breakdown")
        grade_df = pd.DataFrame(
            list(summary["grade_counts"].items()),
            columns=["Severity", "Count"],
        )
        chart_grades = (
            alt.Chart(grade_df)
            .mark_bar()
            .encode(
                x=alt.X("Severity:N", sort=None),
                y=alt.Y("Count:Q"),
                color=alt.Color("Severity:N", legend=None),
            )
            .properties(height=260)
        )
        st.altair_chart(chart_grades, use_container_width=True)

    with col_chart2:
        st.markdown("##### Model Architecture Utilization")
        model_df = pd.DataFrame(
            list(summary["model_usage"].items()),
            columns=["Architecture", "Inferences"],
        )
        chart_models = (
            alt.Chart(model_df)
            .mark_bar(color="#1f77b4")
            .encode(
                x=alt.X("Architecture:N"),
                y=alt.Y("Inferences:Q"),
            )
            .properties(height=260)
        )
        st.altair_chart(chart_models, use_container_width=True)


def run_benchmark_tab() -> None:
    st.subheader("📊 Transfer Learning Model Comparison")

    st.markdown(
        """
        The three deep convolutional neural network architectures are systematically evaluated on the
        stratified validation split ($N = 540$) using clinical and classification metrics:
        """
    )

    # Load any saved metrics from reports
    reports_dir = config.REPORTS_DIR
    records = []
    for arch in config.SUPPORTED_ARCHITECTURES:
        metrics_file = reports_dir / f"{arch}_validation_metrics.json"
        if metrics_file.is_file():
            import json
            with open(metrics_file, "r", encoding="utf-8") as f:
                records.append(json.load(f))

    if records:
        comp_df = pd.DataFrame(records)[
            ["architecture", "accuracy", "quadratic_weighted_kappa", "macro_f1", "weighted_f1", "inference_latency_ms_per_sample"]
        ]
        st.dataframe(comp_df, use_container_width=True)

        chart_comp = (
            alt.Chart(comp_df)
            .mark_bar()
            .encode(
                x=alt.X("architecture:N", title="Backbone Architecture"),
                y=alt.Y("quadratic_weighted_kappa:Q", title="Quadratic Weighted Kappa (QWK)"),
                color=alt.Color("architecture:N", legend=None),
            )
            .properties(height=280)
        )
        st.altair_chart(chart_comp, use_container_width=True)
    else:
        st.info("No saved model evaluation benchmarks found yet under `outputs/reports/`.")

    # Show saved confusion matrices if present
    figures_dir = config.FIGURES_DIR
    cm_images = list(figures_dir.glob("*_confusion_matrix.png"))
    if cm_images:
        st.markdown("##### Generated Confusion Matrices")
        cols = st.columns(len(cm_images))
        for col, img_path in zip(cols, cm_images):
            col.image(str(img_path), caption=img_path.name, use_container_width=True)


def run_ethics_tab() -> None:
    st.subheader("⚖️ Scientific Rigor, Ethical Considerations & Clinical Limitations")
    st.markdown(
        """
        ### 1. Research Scope and Regulatory Status
        - **RetinaInsight** is an experimental academic prototype engineered to explore transfer learning and Explainable AI (Grad-CAM) on retinal photography.
        - It is **not clinically certified** or validated for independent diagnostic deployment.
        - **Never base medical treatment, surgical intervention, or medication changes** on the outputs of this software without formal assessment by a board-certified ophthalmologist or optometrist.

        ### 2. Dataset Specifics and Generalization Bounds
        - Models are trained on the **APTOS 2019 Blindness Detection** dataset (rural India fundus photographs).
        - Variations in retinal cameras (Topcon, Zeiss, Canon), pupil dilation states, lighting artifacts, and patient demographics can impact generalization performance on independent clinical populations.
        - External validation across diverse clinical cohorts (e.g., Messidor-2, EyePACS) is required before drawing real-world conclusions.

        ### 3. Explainability Interpretability Caveat
        - Grad-CAM highlights convolutional feature map activations that correlate mathematically with the classification decision.
        - Saliency heatmaps indicate **influential pixel regions**, but do not constitute medical evidence or prove biological causality.
        """
    )


def main() -> None:
    render_header()
    arch, cam_alpha = render_sidebar()

    tabs = st.tabs([
        "🔬 Screening & Grad-CAM",
        "📋 Screening History",
        "📈 Analytics Dashboard",
        "📊 Model Comparison",
        "⚖️ Ethics & Limitations",
    ])

    with tabs[0]:
        run_screening_tab(architecture=arch, cam_alpha=cam_alpha)
    with tabs[1]:
        run_history_tab()
    with tabs[2]:
        run_analytics_tab()
    with tabs[3]:
        run_benchmark_tab()
    with tabs[4]:
        run_ethics_tab()


if __name__ == "__main__":
    main()
