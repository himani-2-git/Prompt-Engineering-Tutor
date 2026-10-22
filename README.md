# Prompt Engineering Tutor

A RAG chatbot that answers questions about prompt engineering from course notes. It retrieves relevant passages using local BM25 search and streams answers using an LLM.

## Features

- Local BM25 retrieval with automatic acronym expansion (e.g. CoT / Chain of Thought).
- Streaming responses with expandable source citations.
- Multiple answer styles (exam-ready, detailed, beginner).
- Context-aware follow-up question support.
- Multi-provider fallback support (Groq, Gemini, OpenRouter, Cerebras).
- Sliding-window rate limiting.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure API key:
   Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key:
   ```toml
   GROQ_API_KEY = "your-groq-key"
   # or
   GEMINI_API_KEY = "your-gemini-key"
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Configuration

Set keys in `.streamlit/secrets.toml` or as environment variables:

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `CEREBRAS_API_KEY` | Cerebras API key |
| `LLM_CHAIN` | Provider and model chain (e.g., `groq:llama-3.3-70b-versatile,gemini:gemini-flash-latest`) |
| `GLOBAL_HOURLY_LIMIT` | Maximum queries per hour across all sessions (default: 300) |

## Project Structure

```
app.py                 Streamlit web interface
rag/chunking.py        Section-aware document chunking
rag/retriever.py       BM25 retrieval and acronym expansion
rag/llm.py             Streaming LLM client and provider fallback
rag/prompts.py         Prompt templates and formatting
rag/ratelimit.py       Rate limiter
rag/store.py           Chunk index loading and persistence
scripts/               Utility scripts
tests/                 Unit tests
data/                  Course notes and indexed chunks
```

## Testing

Run unit tests with:
```bash
python -m unittest discover -s tests
```
