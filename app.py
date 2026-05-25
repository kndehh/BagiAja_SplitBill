import streamlit as st
import os

st.set_page_config(page_title="Bagi Aja", page_icon="", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 4rem 0 2rem;">
    <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing:-0.02em;">Bagi Aja</h1>
    <p style="color:#64748B; font-size:1.25rem; max-width:560px; margin:1rem auto; line-height:1.6;">
        Upload a receipt, let AI extract every item, then split the bill fairly with your friends.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.page_link("pages/1_AI_OCR.py", label="AI OCR & Preprocessing", use_container_width=True)
    st.page_link("pages/2_Split_Bill.py", label="Split Ur Bill", use_container_width=True)

st.markdown("""
<div style="text-align:center; padding: 3rem 0; color:#64748B; font-size:0.9rem;">
    Made by Group 14
</div>
""", unsafe_allow_html=True)
