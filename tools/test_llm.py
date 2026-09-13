"""
test_llm.py

Simple manual test for llm_client.py — confirms we can talk to
Qwen3 via Ollama.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tools.llm_client import ask_llm


def main():
    prompt = "Say hello in one short sentence and tell me you're Qwen3 running locally."
    print(f"Sending prompt: {prompt}\n")

    reply = ask_llm(prompt)

    print("Model reply:")
    print(reply)


if __name__ == "__main__":
    main()