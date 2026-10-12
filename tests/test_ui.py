"""
Unit tests for UI components, charts, and report generation in src.ui.
"""
import numpy as np
import plotly.graph_objects as go
from src.ui.styles import CUSTOM_CSS
from src.ui.components import plot_risk_gauge, plot_confusion_matrix, plot_roc_curve
from src.ui.report import generate_clinical_report


def test_custom_css_tokens():
    """Verify essential typography and styling classes exist in design system."""
    assert "Plus Jakarta Sans" in CUSTOM_CSS
    assert ".clinical-card" in CUSTOM_CSS
    assert ".system-banner" in CUSTOM_CSS
    assert ".badge-pill" in CUSTOM_CSS


def test_plot_risk_gauge():
    """Verify risk speedometer gauge generates valid Plotly indicator figure."""
    fig_high = plot_risk_gauge(0.85)
    assert isinstance(fig_high, go.Figure)
    assert fig_high.data[0].type == "indicator"
    assert fig_high.data[0].value == 85.0

    fig_low = plot_risk_gauge(0.20)
    assert isinstance(fig_low, go.Figure)
    assert fig_low.data[0].value == 20.0


def test_plot_confusion_matrix():
    """Verify annotated confusion matrix heatmap generates valid Plotly figure."""
    cm = np.array([[25, 4], [3, 27]])
    fig_cm = plot_confusion_matrix(cm)
    assert isinstance(fig_cm, go.Figure)
    assert len(fig_cm.data) > 0
    assert fig_cm.data[0].type == "heatmap"


def test_plot_roc_curve():
    """Verify ROC curve plotly figure generation."""
    fpr = [0.0, 0.1, 0.2, 1.0]
    tpr = [0.0, 0.8, 0.9, 1.0]
    fig_roc = plot_roc_curve(fpr, tpr, 0.898)
    assert isinstance(fig_roc, go.Figure)
    assert len(fig_roc.data) == 2  # Model line and random diagonal


def test_generate_clinical_report():
    """Verify patient clinical screening summary string generator."""
    pred_res = {
        'timestamp': '2026-10-12 18:00:00',
        'patient_summary': {
            'Usia': 58,
            'Gender': 'Pria',
            'Nyeri Dada': 'Typical Angina',
            'Detak Max': '145 bpm',
            'Depresi ST': '1.8 mm',
            'Angina Olahraga': 'Ya',
            'Pembuluh Tersumbat': '1',
            'Thallium Scan': 'Normal'
        }
    }
    report = generate_clinical_report(
        pred_res=pred_res,
        prediction=1,
        probability=0.78,
        rec_text="Konsultasikan segera dengan kardiolog."
    )
    assert "RINGKASAN SKRINING RISIKO PENYAKIT JANTUNG (CDSS)" in report
    assert "Risiko Tinggi Penyakit Jantung" in report
    assert "78.00%" in report
    assert "Typical Angina" in report
    assert "DISCLAIMER MEDIS" in report
