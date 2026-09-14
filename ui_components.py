import streamlit as st

def section(title,kicker=None):
    if kicker: st.caption(kicker.upper())
    st.markdown(f"<h2 class='section-title'>{title}</h2>",unsafe_allow_html=True)
def metric(label,value,sub='',tone='cyan'):
    st.markdown(f"<div class='metric metric-{tone}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",unsafe_allow_html=True)
def risk_badge(verdict):
    tone='critical' if 'CRITICAL' in verdict else 'high' if 'HIGH' in verdict else 'moderate' if 'MODERATE' in verdict else 'low'
    st.markdown(f"<span class='badge badge-{tone}'>{verdict}</span>",unsafe_allow_html=True)
def nav(active):
    items=['HOME','FORENSIC ANALYSIS','CASE FILES','MODEL LAB','SYSTEM INFO']
    cols=st.columns(len(items))
    for c,item in zip(cols,items):
        if c.button(item, key='nav_'+item, use_container_width=True): st.session_state.page=item
    st.markdown(f"<div class='nav-active'>ACTIVE MODULE / {active}</div>",unsafe_allow_html=True)
