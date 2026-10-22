import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.store import build_and_save

if __name__ == "__main__":
    chunks = build_and_save()
    print(f"Saved {len(chunks)} chunks to data/chunks.json")
