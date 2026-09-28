import os
import requests
import streamlit as st

st.set_page_config(page_title="BI RAG Agent", page_icon="📊", layout="wide")
st.title("Business Intelligence RAG Agent")
st.caption("Grounded answers across SQL data, KPI definitions, and business documents.")
question = st.text_input("Ask a business question", "Why did revenue decline in Q3?")
if st.button("Analyze", type="primary"):
    with st.spinner("Retrieving governed evidence..."):
        response = requests.post(f"{os.getenv('API_URL', 'http://localhost:8000')}/ask", json={"question": question}, timeout=30)
        result = response.json()
    st.subheader("Grounded answer")
    st.write(result["answer"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Q2 revenue", f"${result['data']['q2_revenue']:,.0f}")
    c2.metric("Q3 revenue", f"${result['data']['q3_revenue']:,.0f}", f"{result['data']['change_pct']}%")
    c3.metric("Evidence checks", result["evaluation"]["citation_count"])
    st.subheader("Segment contribution")
    st.bar_chart({x["segment"]: x["change"] for x in result["data"]["drivers"]})
    with st.expander("Evidence & citations", expanded=True):
        for citation in result["citations"]:
            st.markdown(f"**{citation['kind'].upper()} · {citation['title']}**  ")
            st.caption(f"{citation['locator']} — {citation.get('excerpt') or ''}")
    with st.expander("Reviewed SQL template"):
        st.code(result["sql"], language="sql")

