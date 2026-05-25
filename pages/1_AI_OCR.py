import streamlit as st
import os
import numpy as np
import json
import pandas as pd
from PIL import Image

from models.ai_engine import (
    load_donut_model, run_donut_ocr, run_tesseract_ocr,
    parsed_to_split_bill, preprocess_receipt, score_parsed_result
)

st.set_page_config(page_title="Bagi Aja", page_icon="", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "../style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_models():
    here = os.path.dirname(os.path.abspath(__file__))
    # HF Spaces root is 1 level up from pages/
    root = os.path.abspath(os.path.join(here, ".."))
    base_path = os.path.join(root, "donut-base-finetuned-cord-v2")
    v6_path = os.path.join(root, "donut-mega-finetuned-final-v6")
    return {
        "base": {
            "path": base_path,
            "model": load_donut_model(base_path, hf_fallback="naver-clova-ix/donut-base-finetuned-cord-v2")
        },
        "v6": {
            "path": v6_path,
            "model": load_donut_model(v6_path, hf_fallback="Kndeh/Finetuned_Donut_V6")
        },
    }

MODELS = get_models()
BASE_PATH = MODELS["base"]["path"]
V6_PATH = MODELS["v6"]["path"]
base_processor, base_model, base_device, base_loaded = MODELS["base"]["model"]
v6_processor, v6_model, v6_device, v6_loaded = MODELS["v6"]["model"]
if not base_loaded:
    st.error(f"❌ Donut base failed to load from: `{BASE_PATH}`")
    st.caption(f"Folder exists: {os.path.isdir(BASE_PATH)} • Device target: {base_device}")
if not v6_loaded:
    st.error(f"❌ Donut v6 final failed to load from: `{V6_PATH}`")
    st.caption(f"Folder exists: {os.path.isdir(V6_PATH)} • Device target: {v6_device}")

# Session state init
for k, v in {
    "donut_base_parsed": None,
    "donut_base_json": None,
    "donut_v6_parsed": None,
    "donut_v6_json": None,
    "tess_parsed": None,
    "tess_json": None,
    "preprocess_steps": [],
    "img_arr": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Hero
status_color = "#99ED71" if base_loaded and v6_loaded else "#FFE566" if base_loaded or v6_loaded else "#FFB3B3"
status_text = f"{'🟢' if base_loaded else '🔴'} Donut base • {'🟢' if v6_loaded else '🔴'} Donut v6 final"
st.markdown(f"""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <h1 style="font-size: 2rem; font-weight: 700;">AI OCR &amp; Preprocessing</h1>
    <p style="color:#64748B;">Upload a receipt and compare Tesseract, Donut base, and Donut v6 final</p>
    <span style="background:{status_color}; color:#000; padding:4px 12px; border-radius:999px; font-weight:600; font-size:0.8rem;">{status_text}</span>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload receipt image", type=["png", "jpg", "jpeg", "webp"])

img_arr = None
if uploaded is not None:
    img_arr = np.array(Image.open(uploaded).convert("RGB"))
    st.session_state.img_arr = img_arr
    st.image(img_arr, use_container_width=True)
elif st.session_state.img_arr is not None:
    img_arr = st.session_state.img_arr
    st.image(img_arr, use_container_width=True)

if img_arr is not None:
    with st.expander("⚙️ Preprocessing Pipeline", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            blur_type = st.selectbox("Blur Algorithm", ["Gaussian", "Median", "Box"], index=0, key="blur_type")
            blur_kernel = st.slider("Kernel Size", 1, 15, 5, step=2, key="blur_kernel")
        with c2:
            blur_sigma = st.slider("Sigma", 0.0, 10.0, 0.0, step=0.5, key="blur_sigma")
            thresh_block = st.slider("Threshold Block", 3, 51, 11, step=2, key="thresh_block")
        with c3:
            thresh_c = st.slider("Threshold C", -10, 30, 2, step=1, key="thresh_c")
        a1, a2, a3 = st.columns(3)
        with a1:
            en_clahe = st.toggle("CLAHE", False, key="clahe")
            en_denoise = st.toggle("Denoise", False, key="denoise")
        with a2:
            en_bilateral = st.toggle("Bilateral", False, key="bilateral")
            en_sharpen = st.toggle("Sharpen", False, key="sharpen")
        with a3:
            en_morph = st.toggle("Morphology", False, key="morph")
            en_perspective = st.toggle("Perspective Fix", False, key="persp")

        st.divider()
        feed_preprocessed = st.toggle(
            "🎯 Feed preprocessed image to Donut",
            value=True,
            key="feed_preprocessed",
            help="ON: Donut sees the thresholded/binary image. OFF: Donut sees the auto-cropped raw RGB receipt."
        )

    def _kwargs():
        return dict(
            blur_type=blur_type, blur_kernel=int(blur_kernel), blur_sigma=float(blur_sigma),
            thresh_block=int(thresh_block), thresh_c=int(thresh_c),
            enable_bilateral=en_bilateral, enable_denoise=en_denoise,
            enable_morph=en_morph, enable_sharpen=en_sharpen,
            enable_clahe=en_clahe, enable_perspective=en_perspective
        )

    b1, b2, b3, b4 = st.columns([1, 1, 1, 1])
    with b1:
        if st.button("👁️ Show Preprocessing Steps", use_container_width=True):
            with st.spinner("Preprocessing..."):
                steps, _ = preprocess_receipt(img_arr, **_kwargs())
                st.session_state.preprocess_steps = steps
    with b2:
        if st.button("🍩 Extract — Donut Base", use_container_width=True, type="primary"):
            with st.spinner("Running Donut base..."):
                try:
                    steps, jout, parsed = run_donut_ocr(img_arr, base_processor, base_model, base_device, base_loaded, use_preprocessed=feed_preprocessed, **_kwargs())
                    st.session_state.preprocess_steps = steps
                    st.session_state.donut_base_parsed = parsed
                    st.session_state.donut_base_json = jout
                    st.toast("✅ Donut base extraction complete!", icon="🍩")
                except Exception as e:
                    st.error(f"Donut base failed: {e}")
    with b3:
        if st.button("🍩 Extract — Donut v6 Final", use_container_width=True, type="primary"):
            with st.spinner("Running Donut v6 final..."):
                try:
                    steps, jout, parsed = run_donut_ocr(img_arr, v6_processor, v6_model, v6_device, v6_loaded, use_preprocessed=feed_preprocessed, **_kwargs())
                    st.session_state.preprocess_steps = steps
                    st.session_state.donut_v6_parsed = parsed
                    st.session_state.donut_v6_json = jout
                    st.toast("✅ Donut v6 extraction complete!", icon="🍩")
                except Exception as e:
                    st.error(f"Donut v6 failed: {e}")
    with b4:
        if st.button("🔤 Extract — Tesseract", use_container_width=True, type="primary"):
            with st.spinner("Running Tesseract OCR..."):
                try:
                    steps, jout, parsed = run_tesseract_ocr(img_arr, **_kwargs())
                    st.session_state.preprocess_steps = steps
                    st.session_state.tess_parsed = parsed
                    st.session_state.tess_json = jout
                    st.toast("✅ Tesseract extraction complete!", icon="🔤")
                except Exception as e:
                    st.error(f"Tesseract failed: {e}")

    if st.session_state.preprocess_steps:
        st.subheader("Preprocessing Pipeline Visualization")
        cols = st.columns(min(len(st.session_state.preprocess_steps), 6))
        for i, (im, cap) in enumerate(st.session_state.preprocess_steps):
            with cols[i % 6]:
                st.image(im, caption=cap, use_container_width=True)

    # ── Results ──
    results = [
        ("🍩 Donut Base", "donut_base", st.session_state.donut_base_parsed, st.session_state.donut_base_json),
        ("🍩 Donut v6 Final", "donut_v6", st.session_state.donut_v6_parsed, st.session_state.donut_v6_json),
        ("🔤 Tesseract OCR", "tess", st.session_state.tess_parsed, st.session_state.tess_json),
    ]
    available_results = [(label, key, parsed, raw) for label, key, parsed, raw in results if parsed is not None or raw is not None]

    def use_result(parsed):
        df, tax, svc, disc, tot = parsed_to_split_bill(parsed)
        st.session_state.items_df = df
        st.session_state.tax = float(tax)
        st.session_state.svc = float(svc)
        st.session_state.disc = float(disc)
        st.session_state.total = float(tot)
        st.session_state.editor_refresh = st.session_state.get("editor_refresh", 0) + 1
        st.switch_page("pages/2_Split_Bill.py")

    def render_result_card(label, key, parsed, raw):
        st.markdown(f"#### {label}")
        if parsed:
            items = parsed.get("items", [])
            st.caption(f"{len(items)} item(s) extracted")
            if items:
                st.dataframe(
                    [{"Name": i["item_name"], "Qty": i["item_quantity"], "Price": i["item_price"]} for i in items],
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning(f"No items were extracted by {label}.")
        else:
            st.info(f"Run {label} extraction to see results.")
        if raw:
            json_str = json.dumps(raw, indent=2, default=str)
            st.code(json_str, language="json")
        if st.button(f"✅ Use {label} Result", key=f"use_{key}", use_container_width=True, disabled=parsed is None):
            use_result(parsed)

    if available_results:
        st.divider()
        st.subheader("🔬 OCR Results")

        cols = st.columns(3)
        for idx, result in enumerate(results):
            with cols[idx]:
                render_result_card(*result)

    # ── Comparison Panel ──
    scored_results = []
    for label, key, parsed, _ in results:
        if parsed is not None:
            score, detail = score_parsed_result(parsed)
            scored_results.append({
                "label": label,
                "key": key,
                "parsed": parsed,
                "score": score,
                "detail": detail,
            })

    if len(scored_results) >= 2:
        st.divider()
        scored_results = sorted(scored_results, key=lambda x: x["score"], reverse=True)
        best = scored_results[0]
        second = scored_results[1]
        gap = best["score"] - second["score"]
        conf = "high" if gap >= 10 else ("moderate" if gap >= 4 else "low")

        conf_color = {"high": "#99ED71", "moderate": "#FFE566", "low": "#FFB3B3"}[conf]
        rec_label = f"{best['label']} is more accurate" if gap > 0 else "🤝 Top engines tied"

        st.markdown(f"""
<div style="text-align:center; padding:1rem 0 0.5rem;">
    <h3 style="margin-bottom:0.25rem;">🆚 Accuracy Comparison</h3>
    <span style="background:{conf_color}; color:#000; padding:5px 16px; border-radius:999px;
                font-weight:700; font-size:0.95rem;">{rec_label}</span>
    <span style="color:#64748B; font-size:0.8rem; margin-left:8px;">Confidence: {conf}</span>
</div>
""", unsafe_allow_html=True)

        score_max = 50
        metric_cols = st.columns(len(scored_results))
        for idx, result in enumerate(scored_results):
            with metric_cols[idx]:
                detail = result["detail"]
                st.metric(result["label"], f"{result['score']} / {score_max}")
                st.caption(
                    f"Items: {detail.get('items_found', 0)} • "
                    f"Sum accuracy: {detail.get('sum_accuracy_pct', 0)}%"
                )

        with st.expander("📊 Detailed scoring breakdown", expanded=False):
            rows = [
                {"Metric": "Items found"},
                {"Metric": "Total detected"},
                {"Metric": "Items sum"},
                {"Metric": "Sum accuracy (%)"},
                {"Metric": "Name quality (%)"},
                {"Metric": "Total score"},
            ]
            for result in scored_results:
                detail = result["detail"]
                rows[0][result["label"]] = detail.get("items_found", 0)
                rows[1][result["label"]] = detail.get("total_detected", 0)
                rows[2][result["label"]] = detail.get("items_sum", 0)
                rows[3][result["label"]] = detail.get("sum_accuracy_pct", 0)
                rows[4][result["label"]] = detail.get("name_quality_pct", 0)
                rows[5][result["label"]] = result["score"]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button(f"✅ Use Recommended — {best['label']}", key=f"use_recommended_{best['key']}", use_container_width=True, type="primary"):
            use_result(best["parsed"])
