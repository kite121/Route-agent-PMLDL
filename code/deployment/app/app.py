"""Streamlit interface for the FastAPI semantic tool router."""

from __future__ import annotations

import os

import streamlit as st

from api_client import ApiClient, ApiClientError, Prediction


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
MAX_TOP_K = 15


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> ApiClient:
    """Keep a lightweight HTTP client across Streamlit reruns."""

    return ApiClient(base_url=base_url)


def render_prediction(prediction: Prediction) -> None:
    """Render the API response without performing any model inference locally."""

    if prediction.decision == "route":
        st.success(f"Selected tool: `{prediction.tool}`")
    else:
        st.info("No available tool matches this request.")

    st.metric("Best similarity score", f"{prediction.score:.4f}")
    st.caption(
        f"Decision: `{prediction.decision}` · Model version: `{prediction.model_version}`"
    )
    st.subheader("Top-k tool candidates")
    st.dataframe(
        [
            {"tool": candidate.tool, "similarity_score": candidate.score}
            for candidate in prediction.top_k
        ],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(page_title="Route Agent", page_icon="🧭", layout="centered")
    st.title("Route Agent")
    st.write("Send a request to FastAPI and receive a semantic tool-routing decision.")

    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    st.caption(f"FastAPI endpoint: `{api_base_url}`")
    client = get_api_client(api_base_url)

    text = st.text_area(
        "User request",
        placeholder="For example: Set an alarm for 7 tomorrow morning",
    )
    top_k = st.slider("Number of alternative tools", min_value=1, max_value=MAX_TOP_K, value=3)

    if st.button("Route request", type="primary"):
        if not text.strip():
            st.warning("Enter a request before routing it.")
            return
        try:
            with st.spinner("Requesting prediction from FastAPI..."):
                prediction = client.predict(text=text, top_k=top_k)
        except ApiClientError as error:
            st.error(f"FastAPI request failed: {error}")
            return
        render_prediction(prediction)


if __name__ == "__main__":
    main()
