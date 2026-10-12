"""
Interactive Plotly visualizations for Clinical Decision Support System.
Includes Risk Speedometer Gauge, SHAP Waterfall/Bar, Confusion Matrix, and ROC Curve.
"""
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap


def plot_risk_gauge(probability: float) -> go.Figure:
    """
    Builds a clinical risk speedometer gauge with 3 triage color zones.
    - Low Risk (0 - 40%): Green
    - Borderline (40 - 60%): Yellow
    - High Risk (60 - 100%): Red
    """
    gauge_color = "#DC2626" if probability >= 0.5 else "#059669"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Tingkat Probabilitas Risiko Kardiovaskular", 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 38, 'color': gauge_color, 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': gauge_color, 'thickness': 0.28},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "rgba(128,128,128,0.2)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_shap_summary(model, scaler, input_data: List, feature_names: List[str]) -> Optional[go.Figure]:
    """
    Generates an interactive Plotly horizontal bar chart for patient-level SHAP values.
    Attributes risk positive (Red) vs protective (Green) clinical factors.
    """
    explainer = shap.TreeExplainer(model)
    df_input = pd.DataFrame([input_data], columns=feature_names)
    scaled_raw = scaler.transform(df_input)
    std_data = pd.DataFrame(scaled_raw, columns=feature_names)
    shap_values = explainer.shap_values(std_data)

    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]
    sv = np.ravel(sv)

    readable_feature_names = [
        'Nyeri Dada (cp)', 'Detak Jantung Max (thalach)', 'Kemiringan ST (slope)',
        'Depresi ST EKG (oldpeak)', 'Angina Olahraga (exang)', 'Pembuluh Tersumbat (ca)',
        'Thallium Scan (thal)', 'Gender (sex)', 'Usia (age)'
    ]

    df_shap_plot = pd.DataFrame({
        'Fitur': readable_feature_names,
        'SHAP_Value': sv,
        'Dampak': ['Meningkatkan Risiko' if v > 0 else 'Menurunkan Risiko' for v in sv],
        'Nilai_Abs': np.abs(sv)
    }).sort_values(by='SHAP_Value', ascending=True)

    color_map = {'Meningkatkan Risiko': '#DC2626', 'Menurunkan Risiko': '#059669'}

    fig = px.bar(
        df_shap_plot,
        x='SHAP_Value',
        y='Fitur',
        color='Dampak',
        color_discrete_map=color_map,
        orientation='h',
        title="Atribusi Faktor Risiko Pasien (Shapley Additive exPlanations)",
        labels={'SHAP_Value': 'Pengaruh terhadap Log-Odds Risiko', 'Fitur': 'Parameter Klinis'}
    )
    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='rgba(128,128,128,0.4)'),
        yaxis=dict(title=''),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_model_comparison(df_comp: pd.DataFrame) -> go.Figure:
    """Renders grouped bar chart comparing Accuracy, Recall, and ROC-AUC across algorithms."""
    fig = px.bar(
        df_comp,
        x="Model",
        y=["Accuracy", "Recall", "AUC-ROC"],
        barmode="group",
        title="Perbandingan Metrik Utama antar Algoritma (MLflow Tracking)",
        color_discrete_sequence=['#1E3A8A', '#059669', '#DC2626']
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Renders an interactive annotated Plotly Heatmap for holdout test confusion matrix.
    Includes exact patient counts and percentages per quadrant.
    """
    x_labels = ['Prediksi Sehat (0)', 'Prediksi Sakit (1)']
    y_labels = ['Aktual Sehat (0)', 'Aktual Sakit (1)']
    total_samples = np.sum(cm)

    annotations = []
    for i, row in enumerate(cm):
        for j, val in enumerate(row):
            pct = (val / total_samples) * 100
            annotations.append(dict(
                x=x_labels[j], y=y_labels[i],
                text=f"<b>{val} Pasien</b><br>({pct:.1f}%)",
                showarrow=False,
                font=dict(color="white" if val > np.max(cm) / 2 else "#1E293B", size=14)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=x_labels, y=y_labels,
        colorscale='Blues', showscale=False
    ))
    fig.update_layout(
        annotations=annotations,
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_roc_curve(fpr, tpr, roc_auc: float) -> go.Figure:
    """Renders an interactive ROC Curve with the optimal threshold area."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#3B82F6', width=3),
        name=f"Random Forest (AUC = {roc_auc:.3f})"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='#94A3B8', dash='dash'),
        name="Random Chance"
    ))
    fig.update_layout(
        height=320,
        xaxis=dict(title="False Positive Rate (1 - Spesifisitas)", showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(title="True Positive Rate (Sensitivitas)", showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(x=0.55, y=0.1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_feature_importance(df_feat: pd.DataFrame) -> go.Figure:
    """Renders horizontal bar chart for Random Forest Gini Feature Importance."""
    fig = px.bar(
        df_feat,
        x="Importance",
        y="Feature",
        orientation='h',
        color="Importance",
        color_continuous_scale="Viridis",
        title="Random Forest Gini Feature Importance (MDI)",
        labels={'Importance': 'Tingkat Kepentingan Relatif (0-1)', 'Feature': 'Parameter Medis'}
    )
    fig.update_layout(
        height=380,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_eda_scatter(df_raw: pd.DataFrame) -> go.Figure:
    """Renders scatter plot of Patient Age vs Max Heart Rate with risk coloring."""
    fig = px.scatter(
        df_raw,
        x="age",
        y="thalach",
        color="Status",
        color_discrete_map={'Normal / Sehat (0)': '#059669', 'Penyakit Jantung (1)': '#DC2626'},
        labels={'age': 'Usia (Tahun)', 'thalach': 'Detak Jantung Maksimal (bpm)'},
        hover_data=['cp', 'oldpeak']
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_eda_bar(df_raw: pd.DataFrame) -> go.Figure:
    """Renders grouped bar chart of Exercise Induced Angina vs Disease Status."""
    df_exang = df_raw.groupby(['exang', 'Status']).size().reset_index(name='Jumlah Pasien')
    df_exang['Exercise Angina'] = df_exang['exang'].map({0: 'Tidak Ada Angina Olahraga', 1: 'Ada Angina Olahraga'})

    fig = px.bar(
        df_exang,
        x="Exercise Angina",
        y="Jumlah Pasien",
        color="Status",
        barmode="group",
        color_discrete_map={'Normal / Sehat (0)': '#059669', 'Penyakit Jantung (1)': '#DC2626'}
    )
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
