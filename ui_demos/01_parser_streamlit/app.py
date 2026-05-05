"""Streamlit UI wrapper around the email parser.

Reuses 01_parser/parser.py and 01_parser/models.py unchanged.

Run from the lesson root:
    streamlit run ui_demos/01_parser_streamlit/app.py --browser.gatherUsageStats false
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "01_parser"))

import streamlit as st
from parser import parse_email_to_crm
from models import CRMEntry, Intent


SAMPLE_REFUND = """Hey, I'm super frustrated. I bought your software last week and it's
crashing every time I open the dashboard. I want my money back ASAP! This is a disaster.
My name is John Doe, by the way. Fix this or refund me."""

SAMPLE_SALES = """Hi - I'm Priya Sharma, head of growth at Acme Corp. We're evaluating tooling
for our 200-person engineering org and your agentic AI platform looks like a strong fit.
Could someone send pricing for the Enterprise tier and book a demo for next Tuesday?"""

SAMPLE_SUPPORT = """My name is Carlos Mendez. I've been a customer for two years.
The OAuth login flow is broken on the iOS app since the v3.4 update - none of my team
can sign in. We have a release freeze next week, so this is urgent."""

SAMPLE_PARTNERSHIP = """Hello, I'm Jenna Kim, head of partnerships at BetaTech. We're
building a vertical-AI platform for healthcare and would love to explore an OEM /
co-marketing arrangement with your team. Open to a 30-minute intro call next week?"""

PROFILES = {
    "OpenAI GPT-4o-mini (cloud)": {
        "OFFLINE_MODE": "false",
        "LLM_MODEL": "gpt-4o-mini",
    },
    "Anthropic Claude 3.5 Sonnet (cloud)": {
        "OFFLINE_MODE": "false",
        "LLM_MODEL": "claude-3-5-sonnet-20241022",
    },
    "Ollama Llama 3.1 (offline)": {
        "OFFLINE_MODE": "true",
        "OLLAMA_MODEL": "ollama_chat/llama3.1:latest",
    },
}

INTENT_COLORS = {
    Intent.SALES: "#10b981",
    Intent.SUPPORT: "#3b82f6",
    Intent.REFUND: "#ef4444",
    Intent.PARTNERSHIP: "#8b5cf6",
}


st.set_page_config(page_title="Email-to-CRM Parser", layout="wide")

if "email_text" not in st.session_state:
    st.session_state.email_text = ""
if "parsed" not in st.session_state:
    st.session_state.parsed = None
if "parse_error" not in st.session_state:
    st.session_state.parse_error = None


def load_sample(text):
    st.session_state.email_text = text
    st.session_state.parsed = None
    st.session_state.parse_error = None


with st.sidebar:
    st.markdown("## Settings")
    active_profile = st.selectbox(
        "Model provider",
        options=list(PROFILES.keys()),
        help="Same parser code, different LLM provider",
    )

    st.markdown("---")
    st.markdown("## Sample emails")
    st.button("Refund Request", use_container_width=True, on_click=load_sample, args=(SAMPLE_REFUND,))
    st.button("Sales Lead", use_container_width=True, on_click=load_sample, args=(SAMPLE_SALES,))
    st.button("Support Ticket", use_container_width=True, on_click=load_sample, args=(SAMPLE_SUPPORT,))
    st.button("Partnership Inquiry", use_container_width=True, on_click=load_sample, args=(SAMPLE_PARTNERSHIP,))


st.title("Email-to-CRM Parser")
st.markdown(f"Paste a messy email. Get a Pydantic-validated CRM entry back. Active model: **{active_profile}**")

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Email")
    email_text = st.text_area(
        label="email",
        height=400,
        label_visibility="collapsed",
        placeholder="Paste an email here, or click a sample on the left...",
        key="email_text",
    )
    parse_clicked = st.button("Parse to CRM", type="primary", use_container_width=True)

    if parse_clicked:
        if not email_text.strip():
            st.warning("Paste some email text first.")
        else:
            for k, v in PROFILES[active_profile].items():
                os.environ[k] = v
            with st.spinner(f"Calling {active_profile}..."):
                try:
                    st.session_state.parsed = parse_email_to_crm(email_text, response_model=CRMEntry)
                    st.session_state.parse_error = None
                except Exception as e:
                    st.session_state.parsed = None
                    st.session_state.parse_error = str(e)


with col_output:
    st.markdown("### CRM Entry")

    if st.session_state.parse_error:
        st.error(f"**Parsing failed on {active_profile}**")
        st.code(st.session_state.parse_error, language="text")

    elif st.session_state.parsed:
        entry = st.session_state.parsed
        intent_color = INTENT_COLORS[entry.intent]

        with st.container(border=True):
            st.markdown(f"**Sender:** {entry.sender_name}")
            st.markdown(
                f"<span style='background:{intent_color};color:white;padding:6px 16px;"
                f"border-radius:999px;font-weight:600;display:inline-block'>{entry.intent.value}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Priority — {entry.priority_score} / 10**")
            st.progress(entry.priority_score / 10)
            st.caption(entry.urgency_explanation)
            st.markdown("**Action items**")
            for item in entry.action_items:
                st.markdown(f"- {item}")

        with st.expander("Raw JSON"):
            st.code(entry.model_dump_json(indent=2), language="json")

    else:
        st.info("Click **Parse to CRM** on the left to see structured output here.")