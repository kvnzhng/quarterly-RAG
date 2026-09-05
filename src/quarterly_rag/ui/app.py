"""The Streamlit page (RAG-014). Run it with `make ui`.

This talks to the API over HTTP and nothing else. It imports `Settings` for the API address
and `render` for the formatting, and it never touches the pipeline: the point of the page is
to show what a client of `POST /ask` can see, so a second path into the pipeline would make
the page prove nothing.

Streamlit re-runs this whole script on every widget interaction, so the answer lives in
`st.session_state` and a model is called only when the button is pressed.
"""

from __future__ import annotations

import httpx
import streamlit as st

from quarterly_rag.config import get_settings
from quarterly_rag.ui.render import (
    as_markdown,
    citation_label,
    figures_to_mark,
    highlight,
    refusal_headline,
    trace_url,
    verdict,
)

EXAMPLES = [
    "What were Apple's total net sales in the third quarter of fiscal 2026?",
    "What share of Apple's fiscal 2025 total net sales came from Services?",
    "How many employees did Nvidia have at the end of fiscal 2026?",
    "What were Microsoft's total net sales in fiscal 2025?",
]


def fetch(api_url: str, question: str, k: int, ticker: str | None) -> dict:
    body = {"question": question, "k": k}
    if ticker:
        body["ticker"] = ticker
    response = httpx.post(f"{api_url.rstrip('/')}/ask", json=body, timeout=180)
    response.raise_for_status()
    return response.json()


def health(api_url: str) -> dict | None:
    try:
        response = httpx.get(f"{api_url.rstrip('/')}/health", timeout=5)
    except httpx.HTTPError:
        return None
    return response.json() if response.status_code == 200 else None


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title="quarterly-RAG", page_icon="📄", layout="wide")
    st.title("quarterly-RAG")
    st.caption(
        "Answers about Apple and Nvidia SEC filings, from the filings. "
        "Every sentence carries a citation that was checked, or the question is refused."
    )

    with st.sidebar:
        st.subheader("Settings")
        api_url = st.text_input("API", value=settings.api_url)
        k = st.slider("Passages retrieved", min_value=1, max_value=20, value=5)
        ticker = st.selectbox("Company", options=["any", "AAPL", "NVDA"])
        status = health(api_url)
        if status is None:
            st.error("The API is not answering. Start it with `make api`.")
        else:
            st.success(f"{status['model']}\n\nprompt v{status['prompt_version']}")
            st.caption("tracing on" if status["tracing"] else "tracing off")

    question = st.text_input("Question", value=EXAMPLES[0])
    st.caption("Try: " + " · ".join(EXAMPLES[1:]))
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving, answering, and checking every sentence"):
            try:
                st.session_state["result"] = fetch(
                    api_url, question, k, None if ticker == "any" else ticker
                )
            except httpx.HTTPStatusError as exc:
                st.session_state["result"] = None
                st.error(f"The API answered {exc.response.status_code}: {exc.response.text}")
            except httpx.HTTPError as exc:
                st.session_state["result"] = None
                st.error(f"Could not reach the API: {exc}")

    result = st.session_state.get("result")
    if result:
        render(result, settings.langfuse_host)


def render(result: dict, langfuse_host: str) -> None:
    if result.get("refusal"):
        render_refusal(result["refusal"])
    elif result.get("answer"):
        render_answer(result["answer"])
    if result.get("trace_id"):
        st.caption(
            f"[trace {result['trace_id'][:12]}]({trace_url(langfuse_host, result['trace_id'])})"
        )


def render_refusal(refusal: dict) -> None:
    st.warning(f"**{refusal_headline(refusal)}.** {as_markdown(refusal['detail'])}")
    if refusal.get("best_chunks"):
        st.subheader("The closest passages, so you can look yourself")
        for hit in refusal["best_chunks"]:
            with st.expander(
                f"{hit['ticker']} {hit['form']} {hit['period_label']}, {hit['section']}"
                f"  ·  score {hit['score']:.3f}"
            ):
                st.write(hit["excerpt"])
                st.caption(f"[the filing on EDGAR]({hit['source_url']})")


def render_answer(answer: dict) -> None:
    st.markdown(f"### {as_markdown(answer['prose'] or answer['text'])}")
    level, message = verdict(answer)
    getattr(st, level)(as_markdown(message))

    if answer.get("calculations"):
        st.subheader("Arithmetic, recomputed")
        st.caption(
            "A figure no filing prints is only shown as verified when its operands are in "
            "the passages they cite and the sum comes out the same."
        )
        for calculation in answer["calculations"]:
            mark = "✅" if calculation["verified"] else "⚠️"
            st.markdown(
                f"{mark} `{calculation['raw']}`"
                + ("" if calculation["verified"] else f"  ·  {calculation['reason']}")
            )

    if answer.get("citations"):
        st.subheader("Citations")
        st.caption("Highlighted figures are the ones checked against this passage and found.")
        for citation in answer["citations"]:
            with st.expander(citation_label(citation)):
                figures = figures_to_mark(answer["text"], citation["text"])
                st.markdown(
                    f"<div style='white-space:pre-wrap;font-family:ui-monospace,monospace;"
                    f"font-size:0.85rem'>{highlight(citation['text'], figures)}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"[the filing on EDGAR]({citation['source_url']})")

    st.caption(f"{answer['model']} · prompt v{answer['prompt_version']}")


if __name__ == "__main__":
    main()
