"""
Design system, typography, and styling tokens for the Clinical Decision Support System.
"""
import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Card Elevation & Container Styles */
.clinical-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.25s ease;
}

.clinical-card:hover {
    box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.08);
    border-color: #CBD5E1;
}

/* Top System Health Banner */
.system-banner {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    color: #F8FAFC;
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 24px;
    font-size: 0.88rem;
}

.badge-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.76rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge-danger {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FCA5A5;
}

.badge-success {
    background-color: #D1FAE5;
    color: #065F46;
    border: 1px solid #6EE7B7;
}

.badge-info {
    background-color: #E0F2FE;
    color: #075985;
    border: 1px solid #7DD3FC;
}

.badge-warning {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FCD34D;
}

/* Action Callout Box */
.action-callout {
    border-left: 4px solid #3B82F6;
    background-color: #EFF6FF;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
}
</style>
"""


def apply_custom_styles():
    """Injects medical-grade custom CSS rules into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_system_banner():
    """Renders the top CDSS health and capabilities banner."""
    st.markdown("""
    <div class="system-banner">
        <div>
            <strong>🏥 Clinical Decision Support System (CDSS)</strong> • <em>Cardiovascular Screening Engine</em>
        </div>
        <div>
            <span class="badge-pill badge-info">Random Forest Tuned (AUC 0.898)</span>
            <span class="badge-pill badge-success">SHAP XAI: Active</span>
            <span class="badge-pill badge-warning">Gemini AI / Fallback: Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
