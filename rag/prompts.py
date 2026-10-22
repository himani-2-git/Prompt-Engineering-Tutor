from __future__ import annotations

STYLES = {
    "Exam-ready (short)": (
        "Start with a simple 2-3 sentence definition. Then give 3-5 bullet points with the key ideas. "
        "Keep it to about 120-180 words."
    ),
    "Detailed explanation": (
        "Give a clear, structured explanation in about 250-350 words: a short definition, how it works, "
        "one concrete example, and when to use it."
    ),
    "Explain like I'm new": (
        "Explain as if to a complete beginner, using one everyday analogy and no jargon. "
        "About 120-160 words."
    ),
}
DEFAULT_STYLE = "Exam-ready (short)"

SYSTEM = """You are a friendly university tutor helping a student study prompt engineering.
Answer ONLY from the NOTES provided in the user's message.
- If the notes do not contain the answer, say so plainly ("That isn't covered in the course notes") and, if useful, point to the closest topic that is. Never invent facts.
- Use very simple, clear English, like explaining to a classmate. Rewrite ideas in your own words; do not copy long passages.
- Do not mention section numbers, page numbers, or the words "notes" or "context" in your answer.
- Format with Markdown (short paragraphs, bullet points).
STYLE: {style}"""

OUT_OF_SCOPE = (
    "I couldn't find anything about that in the course notes. I can help with topics like zero-/few-shot prompting, "
    "chain-of-thought, prompt patterns, prompt evaluation and debugging, LLM basics, fine-tuning vs prompting, "
    "and LangChain templates. Try asking about one of those!"
)


def format_context(hits) -> str:
    return "\n\n".join(f"[{i}] {h.chunk['title']}\n{h.chunk['text']}" for i, h in enumerate(hits, 1))


def build_messages(question: str, hits, history: list[dict], style: str, max_history: int = 6) -> list[dict]:
    system = SYSTEM.format(style=STYLES.get(style, STYLES[DEFAULT_STYLE]))
    user = f"NOTES:\n{format_context(hits)}\n\nSTUDENT QUESTION: {question}"
    past = [{"role": m["role"], "content": m["content"]} for m in history[-max_history:]]
    return [{"role": "system", "content": system}, *past, {"role": "user", "content": user}]


def retrieval_query(question: str, history: list[dict]) -> str:
    if len(question.split()) < 5:
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev:
            return f"{prev[-1]} {question}"
    return question
