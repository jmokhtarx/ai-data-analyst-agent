"""
llm_client.py

Step 10 of the AI Data Analyst Agent project.

Switched from local Ollama to Groq's free cloud API for speed.
Groq runs open-weight models (Llama, Qwen, etc.) on custom hardware
built for fast inference - much faster than CPU-only local inference.

The API is OpenAI-compatible, so this looks similar to any OpenAI-style
client, just pointed at Groq's endpoint.
"""

import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# IMPORTANT: put your real key here, or better, set it as an environment
# variable named GROQ_API_KEY and leave this as os.environ.get(...).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

MODEL_NAME = "openai/gpt-oss-20b"  # fast, solid general-purpose model on Groq's free tier


def ask_llm(prompt: str) -> str:
    """
    Send a prompt to the LLM via Groq's API and return its reply.

    Parameters
    ----------
    prompt : str
        The user/system message to send to the model.

    Returns
    -------
    str
        The model's text response.

    Raises
    ------
    RuntimeError
        If the API key is missing or Groq returns an error.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "No Groq API key set. Set the GROQ_API_KEY environment variable "
            "or paste your key into tools/llm_client.py"
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq returned status {response.status_code}: {response.text}"
        )

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected response shape from Groq: {data}") from e