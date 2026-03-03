# streamlit_app.py
import streamlit as st

st.set_page_config(
    page_title="Metocean & Operability",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Metocean & Operability Explorer")
st.markdown(
    """
Welcome! Use the sidebar to open **Metocean** (maps of mean Hs/Tp) or other pages.

**Tip:** The Metocean page reads the small global 3°×3° scatter file (Hs/Tp probabilities)
you generated and computes statistics on the fly—so it’s very fast.
"""
)