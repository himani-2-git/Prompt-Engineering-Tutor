import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import build_chunks
from rag.llm import AllProvidersFailed, ChatStream, Target, load_chain
from rag.prompts import build_messages, retrieval_query
from rag.ratelimit import SlidingWindowLimiter
from rag.retriever import Retriever, extract_acronyms, tokenize
from rag.store import NOTES

CHUNKS = [c.to_dict() for c in build_chunks(NOTES.read_text(encoding="utf-8"))]
R = Retriever(CHUNKS)


class ChunkingTests(unittest.TestCase):
    def test_many_reasonably_sized_chunks(self):
        self.assertGreater(len(CHUNKS), 50)
        self.assertTrue(all(len(c["text"]) <= 1400 for c in CHUNKS))

    def test_split_words_repaired_and_toc_dropped(self):
        text = " ".join(c["text"] for c in CHUNKS)
        self.assertIn("pronouns", text)
        self.assertNotIn(". . . .", text)
        self.assertNotIn("\\n", text)


class RetrievalTests(unittest.TestCase):
    def top_titles(self, q, k=3):
        return [h.chunk["title"] for h in R.search(q, k)]

    def test_topic_lookup(self):
        self.assertIn("2.1.2 Zero-Shot Prompting", self.top_titles("what is zero-shot prompting"))
        self.assertIn("2.2.4 Socratic Prompt Pattern", self.top_titles("socratic prompt pattern"))

    def test_acronym_expansion(self):
        self.assertIn("3.0.1 Chain of Thought Prompting", self.top_titles("explain CoT"))
        self.assertEqual(extract_acronyms("Chain of Thought (CoT) helps")["cot"], tokenize("Chain of Thought"))

    def test_no_match_returns_nothing(self):
        self.assertEqual(R.search("capital of France"), [])
        self.assertEqual(R.search("hi"), [])

    def test_followup_uses_previous_question(self):
        hist = [{"role": "user", "content": "What is few-shot prompting?"}]
        self.assertIn("few-shot", retrieval_query("and an example?", hist))


class PromptTests(unittest.TestCase):
    def test_messages_shape(self):
        hits = R.search("zero-shot", 2)
        msgs = build_messages("What is zero-shot?", hits, [{"role": "user", "content": "x"}] * 10, "Exam-ready (short)")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertLessEqual(len(msgs), 8)


class RateLimitTests(unittest.TestCase):
    def test_window(self):
        lim = SlidingWindowLimiter(2, 10)
        self.assertTrue(lim.allow("a", now=0))
        self.assertTrue(lim.allow("a", now=1))
        self.assertFalse(lim.allow("a", now=2))
        self.assertTrue(lim.allow("a", now=12))
        self.assertTrue(lim.allow("b", now=2))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        if self.path.startswith("/bad"):
            self.send_response(429)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ("Hello", " ", "wörld"):
            self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": word}}]})}\n\n'.encode())
        self.wfile.write(b'data: {"choices": []}\n\ndata: [DONE]\n\n')


class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_failover_and_streaming(self):
        chain = [Target("groq", "m1", self.base + "/bad", "k"), Target("gemini", "m2", self.base + "/ok", "k")]
        s = ChatStream([{"role": "user", "content": "hi"}], chain)
        self.assertEqual("".join(s.tokens()), "Hello wörld")
        self.assertEqual(s.used.provider, "gemini")
        self.assertIn("rate limit", s.errors[0])

    def test_all_fail(self):
        s = ChatStream([], [Target("groq", "m", self.base + "/bad", "k")])
        with self.assertRaises(AllProvidersFailed):
            list(s.tokens())

    def test_load_chain_skips_missing_keys(self):
        keys = {"GROQ_API_KEY": "x"}
        chain = load_chain(None, keys.get)
        self.assertTrue(chain and all(t.provider == "groq" for t in chain))
        self.assertEqual(load_chain("bogus:model", keys.get), [])


if __name__ == "__main__":
    unittest.main()
