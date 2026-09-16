"""Streamlit interface for the FastAPI semantic tool router."""

from __future__ import annotations

import os

import streamlit as st

from api_client import ApiClient, ApiClientError, Prediction, ToolDefinition


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


def tool_rows(tools: list[ToolDefinition]) -> list[dict[str, str]]:
    """Format API-provided tool metadata for the visible registry table."""

    return [
        {
            "tool": tool.name,
            "description": tool.description,
            "arguments": ", ".join(tool.arguments),
        }
        for tool in tools
    ]


def main() -> None:
    st.set_page_config(page_title="Route Agent", page_icon="🧭", layout="centered")
    st.title("Route Agent")
    st.write("Send a request to FastAPI and receive a semantic tool-routing decision.")

    api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    st.caption(f"FastAPI endpoint: `{api_base_url}`")
    client = get_api_client(api_base_url)

    try:
        registry = client.tools()
    except ApiClientError as error:
        st.error(f"Cannot load the tool registry from FastAPI: {error}")
        return

    tool_names = [tool.name for tool in registry.tools]
    st.subheader("Available tools")
    st.caption(f"Registry packaged with model `{registry.model_version}`")
    st.dataframe(tool_rows(registry.tools), hide_index=True, width="stretch")
    allowed_tools = st.multiselect(
        "Tools available for this request",
        options=tool_names,
        default=tool_names,
        help="The model routes only among selected tools. Clear the selection to block routing.",
    )

    text = st.text_area(
        "User request",
        placeholder="For example: Set an alarm for 7 tomorrow morning",
    )
    if allowed_tools:
        top_k = st.slider(
            "Number of displayed candidates",
            min_value=1,
            max_value=min(MAX_TOP_K, len(allowed_tools)),
            value=min(3, len(allowed_tools)),
        )
    else:
        st.warning("Select at least one tool before routing a request.")
        top_k = 1

    if st.button("Route request", type="primary", disabled=not allowed_tools):
        if not text.strip():
            st.warning("Enter a request before routing it.")
            return
        try:
            with st.spinner("Requesting prediction from FastAPI..."):
                prediction = client.predict(
                    text=text,
                    top_k=top_k,
                    allowed_tools=allowed_tools,
                )
        except ApiClientError as error:
            st.error(f"FastAPI request failed: {error}")
            return
        render_prediction(prediction)


if __name__ == "__main__":
    main()
