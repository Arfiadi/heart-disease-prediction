"""
UI Package for Heart Disease Clinical Decision Support System.
Exposes styles, reusable visual components, and report generators.
"""
from src.ui.styles import CUSTOM_CSS, apply_custom_styles, render_system_banner
from src.ui.components import (
    plot_risk_gauge,
    plot_shap_summary,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_eda_scatter,
    plot_eda_bar,
)
from src.ui.report import generate_clinical_report

__all__ = [
    "CUSTOM_CSS",
    "apply_custom_styles",
    "render_system_banner",
    "plot_risk_gauge",
    "plot_shap_summary",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "plot_eda_scatter",
    "plot_eda_bar",
    "generate_clinical_report",
]
