import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.llm import PROVIDERS, list_models

if __name__ == "__main__":
    for provider, (_, key_name) in PROVIDERS.items():
        key = os.environ.get(key_name, "").strip()
        if not key:
            print(f"- {provider}: skipped ({key_name} not set)")
            continue
        try:
            print(f"\n{provider}:")
            for m in list_models(provider, key):
                print("   ", m)
        except Exception as e:
            print(f"   failed: {e}")
