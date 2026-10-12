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
    background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
    color: inherit;
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.05);
    transition: all 0.25s ease;
}

.clinical-card:hover {
    box-shadow: 0 8px 24px 0 rgba(0, 0, 0, 0.08);
    border-color: rgba(128, 128, 128, 0.4);
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
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.35);
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.2);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.badge-info {
    background-color: rgba(59, 130, 246, 0.2);
    color: #3B82F6;
    border: 1px solid rgba(59, 130, 246, 0.35);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.2);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

/* Action Callout Box */
.action-callout {
    border-left: 4px solid #3B82F6;
    background-color: rgba(59, 130, 246, 0.1);
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 16px 0;
}

/* Prevent text truncation in Selectbox inputs & Dropdown menus */
div[data-baseweb="select"] {
    width: 100% !important;
}

div[data-baseweb="select"] > div {
    min-height: 44px !important;
    height: auto !important;
    padding-top: 2px !important;
    padding-bottom: 2px !important;
}

div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] [role="button"],
div[data-baseweb="select"] [aria-selected="true"],
div[data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
    overflow: visible !important;
    line-height: 1.35 !important;
}

div[data-baseweb="popover"] {
    max-width: 95vw !important;
}

ul[data-baseweb="menu"],
ul[role="listbox"] {
    min-width: 100% !important;
    width: max-content !important;
    max-width: 580px !important;
}

ul[data-baseweb="menu"] li,
ul[role="listbox"] li,
ul[data-baseweb="menu"] li div,
ul[role="listbox"] li div,
ul[data-baseweb="menu"] [role="option"] {
    white-space: normal !important;
    word-break: normal !important;
    overflow-wrap: break-word !important;
    text-overflow: unset !important;
    overflow: visible !important;
    line-height: 1.4 !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
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
