"""Streamlit app: ask questions about reviews, filtered by the ABSA model's labels.

Run from the repo root:  streamlit run rag/app.py
"""
import re

import streamlit as st

from rag_core import answer, ASPECTS, SENTIMENTS

st.set_page_config(page_title="Review Insights", layout="wide")
st.title("Ask the Reviews")
st.caption("Amazon Electronics review sentences, filtered by the ABSA model's "
           "predicted aspect and sentiment, then summarized by an LLM with citations.")

# ---------- Sidebar: filters ----------
with st.sidebar:
    st.header("Filters")
    aspect = st.selectbox("Aspect", ["any"] + ASPECTS, index=1)        # default: battery
    sentiment = st.selectbox("Sentiment", ["any"] + SENTIMENTS, index=1)  # default: negative
    min_conf = st.slider("Min sentiment confidence", 0.0, 0.95, 0.6, 0.05)
    k = st.slider("Sentences to retrieve", 5, 30, 12)
    provider = st.radio("LLM provider", ["groq", "ollama"])

# ---------- Main area: question ----------
question = st.text_input("Your question",
                         "What do customers complain about with battery life?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching reviews and writing the answer..."):
        result = answer(
            question,
            aspect=None if aspect == "any" else aspect,
            sentiment=None if sentiment == "any" else sentiment,
            min_conf=min_conf,
            k=k,
            provider=provider,
        )

    st.subheader("Answer")
    st.markdown(result["answer"])

    # Which source numbers did the LLM actually cite?
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    used = [s for s in result["sources"] if s["n"] in cited]
    unused = [s for s in result["sources"] if s["n"] not in cited]

    def show(source):
        st.markdown(
            f"**[{source['n']}]** {source['text']}  \n"
            f"<small>{int(source['rating'])}-star review · predicted "
            f"{source['aspect']} / {source['sentiment']}</small>",
            unsafe_allow_html=True,
        )

    st.subheader(f"Cited in the answer ({len(used)})")
    for s in used:
        show(s)

    with st.expander(f"Retrieved but not used ({len(unused)})"):
        for s in unused:
            show(s)