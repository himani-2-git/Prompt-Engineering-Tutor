import os
import streamlit as st

from rag.llm import AllProvidersFailed, ChatStream, load_chain
from rag.prompts import DEFAULT_STYLE, OUT_OF_SCOPE, STYLES, build_messages, retrieval_query
from rag.ratelimit import SlidingWindowLimiter
from rag.retriever import Retriever
from rag.store import load_chunks

st.set_page_config(page_title="Prompt Engineering Tutor", page_icon="🎓", layout="centered")

MAX_QUESTION_CHARS = 600
EXAMPLES = [
    "What is chain-of-thought prompting?",
    "Zero-shot vs few-shot: what's the difference?",
    "Fine-tuning vs prompting: when to use which?",
    "What are the main prompt patterns?",
]


def get_secret(name: str):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever(load_chunks())


@st.cache_resource
def get_global_limiter() -> SlidingWindowLimiter:
    return SlidingWindowLimiter(int(get_secret("GLOBAL_HOURLY_LIMIT") or 300), 3600)


retriever = get_retriever()
chain = load_chain(get_secret("LLM_CHAIN"), get_secret)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_limiter" not in st.session_state:
    st.session_state.session_limiter = SlidingWindowLimiter(15, 600)

with st.sidebar:
    st.header("Settings")
    style = st.radio("Answer style", list(STYLES), index=list(STYLES).index(DEFAULT_STYLE))
    top_k = st.slider("Note passages to use", 2, 6, 4)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"{len(retriever.chunks)} passages indexed from course notes")
    if chain:
        st.caption("Active models:")
        for t in chain:
            st.caption(f"• {t.label}")
    else:
        st.error("No API key configured.")

st.title("Prompt Engineering Tutor")
st.caption("Ask questions based on your course notes.")


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['title']}**")
            st.caption(s["text"][:350] + ("…" if len(s["text"]) > 350 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

pending = None
if not st.session_state.messages:
    st.write("**Suggested questions:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}"):
            pending = q

question = st.chat_input("Ask a question about prompt engineering...") or pending

if question:
    question = question.strip()[:MAX_QUESTION_CHARS]
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = retriever.search(retrieval_query(question, history), k=top_k)
        sources = [h.chunk for h in hits]

        if not st.session_state.session_limiter.allow("me") or not get_global_limiter().allow("all"):
            st.warning("Rate limit exceeded. Please wait a few moments and try again.")
            st.session_state.messages.pop()
        elif not chain:
            st.error("API key is not configured.")
            st.session_state.messages.pop()
        elif not hits:
            st.markdown(OUT_OF_SCOPE)
            st.session_state.messages.append({"role": "assistant", "content": OUT_OF_SCOPE})
        else:
            stream = ChatStream(build_messages(question, hits, history, style), chain)
            try:
                answer = st.write_stream(stream.tokens())
                if stream.used:
                    st.caption(f"Answered by {stream.used.label}")
                render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except AllProvidersFailed:
                st.warning("All configured LLM providers are currently unavailable. Please try again shortly.")
                st.session_state.messages.pop()
