# import streamlit as st
# import sys
# import os
# import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# from models.ai_engine import parsed_to_split_bill

# st.set_page_config(page_title="Bagi Aja - Split Bill", page_icon="", layout="wide")

# css_path = os.path.join(os.path.dirname(__file__), "../style.css")
# if os.path.exists(css_path):
#     with open(css_path, "r") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # Session state init
# for k, v in {
#     "items_df": None,
#     "tax": 0.0,
#     "svc": 0.0,
#     "disc": 0.0,
#     "total": 0.0,
#     "editor_refresh": 0,
# }.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# st.markdown("""
# <div style="text-align:center; padding: 1rem 0 0.5rem;">
#     <h1 style="font-size: 2rem; font-weight: 700;">Split Ur Bill</h1>
#     <p style="color:#64748B;">Assign items to people and calculate the split</p>
# </div>
# """, unsafe_allow_html=True)

# if st.session_state.items_df is None:
#     st.info("No items loaded. Please go to the AI OCR page first.")
#     if st.button("Go to AI OCR", use_container_width=True):
#         st.switch_page("pages/1_AI_OCR.py")
# else:
#     # Display summary
#     c1, c2, c3, c4 = st.columns(4)
#     with c1:
#         st.metric("Subtotal", f"Rp {st.session_state.items_df['item_price'].sum():,.0f}")
#     with c2:
#         st.metric("Tax", f"Rp {st.session_state.tax:,.0f}")
#     with c3:
#         st.metric("Service", f"Rp {st.session_state.svc:,.0f}")
#     with c4:
#         st.metric("Total", f"Rp {st.session_state.total:,.0f}")

#     st.divider()

#     # Editable dataframe
#     st.subheader("Items")
#     df = st.session_state.items_df.copy()
    
#     # Add person assignment column if not exists
#     if 'person' not in df.columns:
#         df['person'] = ''
    
#     edited_df = st.data_editor(
#         df,
#         column_config={
#             "item_name": st.column_config.TextColumn("Item Name"),
#             "item_quantity": st.column_config.NumberColumn("Qty", min_value=1, step=1),
#             "item_price": st.column_config.NumberColumn("Price", format="Rp %.0f"),
#             "person": st.column_config.TextColumn("Assign to Person")
#         },
#         num_rows="dynamic",
#         use_container_width=True,
#         key=f"editor_{st.session_state.editor_refresh}"
#     )

#     # Calculate split
#     if st.button("Calculate Split", use_container_width=True, type="primary"):
#         if edited_df['person'].str.strip().eq('').all():
#             st.warning("Please assign at least one item to a person.")
#         else:
#             # Group by person
#             person_totals = edited_df.groupby('person').agg({
#                 'item_price': 'sum',
#                 'item_quantity': 'sum'
#             }).reset_index()
#             person_totals.columns = ['Person', 'Subtotal', 'Items']
            
#             # Calculate proportional extras
#             total_items_price = edited_df['item_price'].sum()
#             if total_items_price > 0:
#                 person_totals['Tax Share'] = (person_totals['Subtotal'] / total_items_price) * st.session_state.tax
#                 person_totals['Service Share'] = (person_totals['Subtotal'] / total_items_price) * st.session_state.svc
#                 person_totals['Discount Share'] = (person_totals['Subtotal'] / total_items_price) * st.session_state.disc
#                 person_totals['Total'] = person_totals['Subtotal'] + person_totals['Tax Share'] + person_totals['Service Share'] - person_totals['Discount Share']
#             else:
#                 person_totals['Tax Share'] = 0
#                 person_totals['Service Share'] = 0
#                 person_totals['Discount Share'] = 0
#                 person_totals['Total'] = person_totals['Subtotal']

#             st.divider()
#             st.subheader("Split Bill Result")
#             st.dataframe(
#                 person_totals,
#                 column_config={
#                     "Person": st.column_config.TextColumn("Person"),
#                     "Subtotal": st.column_config.NumberColumn("Subtotal", format="Rp %.0f"),
#                     "Items": st.column_config.NumberColumn("Items"),
#                     "Tax Share": st.column_config.NumberColumn("Tax Share", format="Rp %.0f"),
#                     "Service Share": st.column_config.NumberColumn("Service Share", format="Rp %.0f"),
#                     "Discount Share": st.column_config.NumberColumn("Discount Share", format="Rp %.0f"),
#                     "Total": st.column_config.NumberColumn("Total", format="Rp %.0f")
#                 },
#                 use_container_width=True,
#                 hide_index=True
#             )

#     st.divider()
#     if st.button("Back to AI OCR", use_container_width=True):
#         st.switch_page("pages/1_AI_OCR.py")
import streamlit as st
import os
import pandas as pd

from models.ai_engine import calculate_split_bill, assign_all_unassigned, add_item, remove_last_item

st.set_page_config(page_title="Split Bill — Bagi Aja", page_icon="💰", layout="wide")

css_path = os.path.join(os.path.dirname(__file__), "../style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state init
for k, v in {
    "items_df": pd.DataFrame(columns=["Item Name", "Qty", "Price", "Assigned To"]),
    "tax": 0.0, "svc": 0.0, "disc": 0.0, "total": 0.0,
    "tax_pct": False, "svc_pct": False, "disc_pct": False,
    "editor_refresh": 0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

refresh = st.session_state.get("editor_refresh", 0)

st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <h1 style="font-size: 2rem; font-weight: 700;">💰 Split Bill Engine</h1>
    <p style="color:#64748B;">Review items, assign people, and calculate everyone's share</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="color:#64748B; font-size:0.95rem; margin-bottom:1rem;">
    In <strong>Assigned To</strong>, type names separated by commas (e.g. <code>Alice, Bob</code>).
</p>
""", unsafe_allow_html=True)

q1, q2 = st.columns([3, 1])
with q1:
    quick_name = st.text_input("Quick Assign Name", placeholder="e.g. Alice", key="quick_name")
with q2:
    st.markdown("<div style='height:1.8rem;'></div>", unsafe_allow_html=True)
    if st.button("📌 Assign All Unassigned", use_container_width=True):
        st.session_state.items_df = assign_all_unassigned(st.session_state.items_df, quick_name)
        st.session_state.editor_refresh = st.session_state.get("editor_refresh", 0) + 1
        st.rerun()

# ── Add / Remove row controls ──
add_col, rem_col = st.columns([1, 1])
with add_col:
    if st.button("➕ Add Item Row", use_container_width=True):
        st.session_state.items_df = add_item(st.session_state.items_df)
        st.session_state.editor_refresh = st.session_state.get("editor_refresh", 0) + 1
        st.rerun()
with rem_col:
    if st.button("➖ Remove Last Row", use_container_width=True):
        st.session_state.items_df = remove_last_item(st.session_state.items_df)
        st.session_state.editor_refresh = st.session_state.get("editor_refresh", 0) + 1
        st.rerun()

edited_df = st.data_editor(
    st.session_state.items_df,
    column_config={
        "Item Name": st.column_config.TextColumn("Item Name"),
        "Qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
        "Price": st.column_config.NumberColumn("Price", min_value=0.0, step=1000.0, format="%.0f"),
        "Assigned To": st.column_config.TextColumn("Assigned To"),
    },
    use_container_width=True,
    key=f"items_editor_{refresh}"
)
st.session_state.items_df = edited_df

st.subheader("🧾 Receipt Summary")

# Percentage mode toggles
pct1, pct2, pct3 = st.columns(3)
with pct1:
    tax_pct = st.toggle("Tax is %", value=st.session_state.tax_pct, key=f"tax_pct_{refresh}")
with pct2:
    svc_pct = st.toggle("Service is %", value=st.session_state.svc_pct, key=f"svc_pct_{refresh}")
with pct3:
    disc_pct = st.toggle("Discount is %", value=st.session_state.disc_pct, key=f"disc_pct_{refresh}")

s1, s2, s3, s4 = st.columns(4)
with s1:
    tax_label = "Tax (%)" if tax_pct else "Tax (Rp)"
    tax_step = 1.0 if tax_pct else 1000.0
    tax = st.number_input(tax_label, value=float(st.session_state.tax), step=tax_step, format="%.2f" if tax_pct else "%.0f", key=f"inp_tax_{refresh}")
with s2:
    svc_label = "Service Charge (%)" if svc_pct else "Service Charge (Rp)"
    svc_step = 1.0 if svc_pct else 1000.0
    svc = st.number_input(svc_label, value=float(st.session_state.svc), step=svc_step, format="%.2f" if svc_pct else "%.0f", key=f"inp_svc_{refresh}")
with s3:
    disc_label = "Discount (%)" if disc_pct else "Discount (Rp)"
    disc_step = 1.0 if disc_pct else 1000.0
    disc = st.number_input(disc_label, value=float(st.session_state.disc), step=disc_step, format="%.2f" if disc_pct else "%.0f", key=f"inp_disc_{refresh}")
with s4:
    total = st.number_input("Total Amount (Rp)", value=float(st.session_state.total), step=1000.0, format="%.0f", key=f"inp_total_{refresh}")

# Calculate actual amounts from percentages
items_sum = st.session_state.items_df["Price"].astype(float).sum() if not st.session_state.items_df.empty else 0.0
tax_amount = items_sum * (tax / 100.0) if tax_pct else tax
svc_amount = items_sum * (svc / 100.0) if svc_pct else svc
disc_amount = items_sum * (disc / 100.0) if disc_pct else disc

# Show calculated amounts when in percentage mode
if tax_pct or svc_pct or disc_pct:
    calc_cols = st.columns(3)
    if tax_pct:
        with calc_cols[0]:
            st.caption(f"Tax = Rp {tax_amount:,.0f}")
    if svc_pct:
        with calc_cols[1]:
            st.caption(f"Service = Rp {svc_amount:,.0f}")
    if disc_pct:
        with calc_cols[2]:
            st.caption(f"Discount = Rp {disc_amount:,.0f}")

mode = st.radio("Split Strategy", ["After Tax (Proportional)", "Before Tax (Equal Extras)"], horizontal=True)

if st.button("🧮 Calculate Split Bill", type="primary", use_container_width=True):
    df_res, summary = calculate_split_bill(st.session_state.items_df, tax_amount, svc_amount, disc_amount, total, mode)
    st.subheader("📊 Split Results")
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    st.markdown(summary)

st.markdown("""
<div style="text-align:center; padding:2rem 0; color:#64748B; font-size:0.9rem;">
    Made by Group 14
</div>
""", unsafe_allow_html=True)
