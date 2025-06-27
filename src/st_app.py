"""
Streamlit app for Multi Agent System (Research pipeline)
"""

import os
import time
import requests
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Research Pipeline Dashboard", layout="wide")

from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Session state init
if "pipeline_id" not in st.session_state:
    st.session_state.pipeline_id = None

st.title("Multi-Agent Research Pipeline Dashboard")

with st.sidebar:
    st.header("Settings for API")
    st.text_input("API Base URL", value=API_URL, key="api_url_input")
    if st.button("Apply URL"):
        # Update session state, env variable. reset pipeline ID
        os.environ["API_URL"] = st.session_state.api_url_input
        st.session_state.pipeline_id = None
        st.rerun()

API_URL = st.session_state.api_url_input or API_URL

# Tabs for endpoints
tabs = st.tabs(["Start Pipeline", "Active Pipelines", "Check Status", "Results", "Cancel Pipeline"])

with tabs[0]:
    st.header("Research Pipeline")
    topic = st.text_input("Research Topic", "")
    depth = st.selectbox("Depth", ["basic", "standard", "deep"], index=1)
    content_types = st.multiselect("Content Types", ["summary", "report", "presentation", "blog_post"], default=["summary", "report"])
    target_audience = st.selectbox("Target Audience", ["general", "technical", "executive"], index=0)
    max_sources = st.slider("Max Sources", min_value=3, max_value=50, value=10, step=1)

    if st.button("Begin Research"):
        # Build payload
        payload = {
            "topic": topic,
            "depth": depth,
            "content_types": content_types,
            "target_audience": target_audience,
            "max_sources": max_sources
        }
        try:
            # Begin research pipeline
            resp = requests.post(f"{API_URL}/research/start", json=payload)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Started pipeline `{data['pipeline_id']}`")
            st.session_state.pipeline_id = data["pipeline_id"]
            st.session_state.started_at = datetime.fromisoformat(data["created_at"])
            st.session_state.eta = datetime.fromisoformat(data["estimated_completion"])
            st.rerun()
        except Exception as e:
            st.error(f"Error starting pipeline: {e}")

with tabs[1]:
    st.header("Active Pipelines")
    try:
        resp = requests.get(f"{API_URL}/research/active")
        resp.raise_for_status()
        active = resp.json().get("active_pipelines", [])
        if not active:
            st.info("No active pipelines.")
        else:
            for p in active:
                st.write(f"• `{p['pipeline_id']}` — Topic: *{p['topic']}*, Progress: {p['progress']}%, Started: {p['created_at']}")
    except Exception as e:
        st.error(f"Error fetching active pipelines: {e}")

with tabs[2]:
    st.header("Check Pipeline Status")
    pid = st.text_input("Pipeline ID", value=st.session_state.pipeline_id or "")
    if st.button("Refresh Status"):
        st.session_state.pipeline_id = pid
        st.rerun()
    if pid:
        try:
            resp = requests.get(f"{API_URL}/research/status/{pid}")
            if resp.status_code == 404:
                st.error("Pipeline not found.")
            else:
                resp.raise_for_status()
                status = resp.json()
                st.write("**Status:**", status["status"])
                st.write("**Progress:**", f"{status['progress']}%")
                st.write("**Current Agent:**", status.get("current_agent"))
                if status.get("error"):
                    st.error(f"Error: {status['error']}")
        except Exception as e:
            st.error(f"Error checking status: {e}")

with tabs[3]:
    st.header("Fetch Pipeline Results")
    pid = st.text_input("Pipeline ID for Results", value=st.session_state.pipeline_id or "", key="results_pid")
    if st.button("Get Results"):
        try:
            resp = requests.get(f"{API_URL}/research/results/{pid}")
            if resp.status_code == 202:
                st.info("Pipeline is still running. Please try again later.")
            elif resp.status_code == 404:
                st.error("Pipeline not found.")
            else:
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Results for `{data['pipeline_id']}` (exec time: {data['execution_time']}s)")
                # Display LLM results
                st.json(data["results"])
                # Display agent metrics
                if data.get("agent_metrics"):
                    st.subheader("Agent Metrics")
                    st.json(data["agent_metrics"])
        except Exception as e:
            st.error(f"Error fetching results: {e}")

with tabs[4]:
    st.header("Cancel a Running Pipeline")
    pid = st.text_input("Pipeline ID to Cancel", value="", key="cancel_pid")
    if st.button("Cancel Pipeline"):
        try:
            resp = requests.delete(f"{API_URL}/research/cancel/{pid}")
            if resp.status_code == 404:
                st.error("Pipeline not found or already completed.")
            else:
                resp.raise_for_status()
                st.success(resp.json().get("message", "Pipeline cancelled"))
                # If cancelling current session pipeline, clear session state
                if st.session_state.pipeline_id == pid:
                    st.session_state.pipeline_id = None
        except Exception as e:
            st.error(f"Error cancelling pipeline: {e}")
