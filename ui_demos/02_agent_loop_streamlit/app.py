"""Streamlit UI wrapper around the Module 2 ReAct stock-research agent.

Reuses 02_agent_loop/agent.py, tools.py, stock_recommendation.py and
stockRecommendationModels.py unchanged.

Run from this folder:
    cd ui_demos/02_agent_loop_streamlit
    uv run --extra ui streamlit run app.py --browser.gatherUsageStats false
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "02_agent_loop"))

import streamlit as st  # noqa: E402
from agent import run_agent  # noqa: E402
from stock_recommendation import get_stock_recommendation  # noqa: E402
from models import StockRecommendation  # noqa: E402


SAMPLE_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

PROFILES = {
    "Ollama Qwen 2.5 (offline)": {
        "OFFLINE_MODE": "true",
        "OLLAMA_MODEL": "ollama_chat/qwen2.5-coder:latest",
    },
    "Ollama Llama 3.1 (offline)": {
        "OFFLINE_MODE": "true",
        "OLLAMA_MODEL": "ollama_chat/llama3.1:latest",
    },
    "OpenAI GPT-4o-mini (cloud)": {
        "OFFLINE_MODE": "false",
        "LLM_MODEL": "gpt-4o-mini",
    },
    "Anthropic Claude 3.5 Sonnet (cloud)": {
        "OFFLINE_MODE": "false",
        "LLM_MODEL": "claude-3-5-sonnet-20241022",
    },
}

ACTION_COLORS = {
    "Buy": "#10b981",
    "Hold": "#6b7280",
    "Sell": "#ef4444",
}


def action_color(action: str) -> str:
    return ACTION_COLORS.get(action.strip().capitalize(), "#3b82f6")


st.set_page_config(page_title="ReAct Stock Research Agent", layout="wide")

if "symbol" not in st.session_state:
    st.session_state.symbol = ""
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
if "agent_trace" not in st.session_state:
    st.session_state.agent_trace = None
if "error" not in st.session_state:
    st.session_state.error = None


def load_sample(ticker: str) -> None:
    st.session_state.symbol = ticker
    st.session_state.recommendation = None
    st.session_state.agent_trace = None
    st.session_state.error = None


with st.sidebar:
    st.markdown("## Settings")
    active_profile = st.selectbox(
        "Model provider",
        options=list(PROFILES.keys()),
        help="Same agent code, different LLM provider",
    )

    st.markdown("---")
    st.markdown("## Sample tickers")
    for ticker in SAMPLE_TICKERS:
        st.button(ticker, use_container_width=True, on_click=load_sample, args=(ticker,))

    st.markdown("---")
    st.caption(
        "Module 2 of the **Agentic AI for Devs** course. "
        "Reuses `02_agent_loop/` unchanged."
    )


st.title("ReAct Stock Research Agent")
st.markdown(
    f"Type a ticker. The agent fetches the live price, searches for news, "
    f"and returns a structured recommendation. Active model: **{active_profile}**"
)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Ticker")
    symbol = st.text_input(
        label="symbol",
        label_visibility="collapsed",
        placeholder="e.g. AAPL",
        key="symbol",
    )
    run_clicked = st.button("Get Recommendation", type="primary", use_container_width=True)

    if run_clicked:
        if not symbol.strip():
            st.warning("Enter a ticker symbol first.")
        else:
            for k, v in PROFILES[active_profile].items():
                os.environ[k] = v
            ticker = symbol.strip().upper()
            with st.spinner(f"Agent reasoning with {active_profile}..."):
                try:
                    trace_query = (
                        f"Analyze {ticker} stock. Get the current price and latest news. "
                        f"Based on this information, provide a recommendation."
                    )
                    st.session_state.agent_trace = run_agent(trace_query)
                    st.session_state.recommendation = get_stock_recommendation(ticker)
                    st.session_state.error = None
                except Exception as e:
                    st.session_state.recommendation = None
                    st.session_state.error = str(e)

    if st.session_state.agent_trace:
        with st.expander("Agent reasoning trace"):
            st.markdown(st.session_state.agent_trace)


with col_output:
    st.markdown("### Recommendation")
    if st.session_state.error:
        st.error(f"**Agent failed on {active_profile}**")
        st.code(st.session_state.error, language="text")
        st.info(
            "Try switching to a different model in the sidebar — small open-source models "
            "sometimes fail tool-calling or strict Pydantic validation."
        )

    elif st.session_state.recommendation:
        rec: StockRecommendation = st.session_state.recommendation
        color = action_color(rec.action)

        with st.container(border=True):
            top_left, top_right = st.columns([2, 1])
            with top_left:
                st.markdown("**Symbol**")
                st.markdown(f"### {rec.symbol}")
            with top_right:
                st.markdown("**Action**")
                st.markdown(
                    f"<span style='background:{color};color:white;"
                    f"padding:6px 16px;border-radius:999px;font-weight:600;"
                    f"display:inline-block;margin-top:6px'>{rec.action}</span>",
                    unsafe_allow_html=True,
                )

            st.divider()

            raw_score = rec.confidence_score
            confidence_unit = raw_score / 100 if raw_score > 1 else raw_score
            confidence_unit = max(0.0, min(1.0, confidence_unit))
            confidence_pct = int(confidence_unit * 100)
            st.markdown(f"**Confidence — {confidence_pct} / 100**")
            st.progress(confidence_unit)

            st.divider()

            st.markdown("**Rationale**")
            st.markdown(rec.rationale)

            st.divider()

            st.markdown("**Key headlines**")
            if rec.key_headlines:
                for headline in rec.key_headlines:
                    st.markdown(f"- {headline}")
            else:
                st.caption("No headlines surfaced by the agent.")

        with st.expander("Raw JSON"):
            st.code(rec.model_dump_json(indent=2), language="json")

        st.caption(f"Generated with **{active_profile}**")

    else:
        st.info("Enter a ticker on the left and click **Get Recommendation**.")