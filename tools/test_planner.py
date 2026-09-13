"""
planner.py

Step 5 of the AI Data Analyst Agent project.

This is the first real "agent" component: the Analysis Planner.

Instead of us calling analyze_dataset() / create_chart() / profile_dataset()
manually, we now ask the LLM to DECIDE which tool is needed to answer a
user's question, and with what arguments.

Important: the LLM does NOT do any calculation here. It only picks a tool
and fills in arguments. The actual math still happens in Python (the tool
functions themselves). This file just handles the "decision" step.

Step 6 adds explain_result(): once a tool has actually run and produced
real numbers, this asks the LLM to explain those numbers in plain language,
directly answering the user's original question.
"""

import json
import re

from tools.llm_client import ask_llm


TOOLS_DESCRIPTION = """
You have access to these tools:

1. profile_dataset
   - Purpose: get basic structure of the dataset (row/column counts, column types, missing values, duplicates).
   - Arguments: none.

2. analyze_dataset
   - Purpose: get descriptive statistics for every column (mean/median/min/max for numbers, top values for categories, date ranges for dates).
   - Arguments: none.

3. create_chart
   - Purpose: create a chart and save it as an image.
   - Arguments:
     - chart_type: one of "bar", "line", "histogram"
     - x_column: name of the column for the x-axis
     - y_column: name of the column for the y-axis (required for bar and line, omit for histogram)
"""

PLANNER_INSTRUCTIONS = """
You are a planning module for a data analysis agent. You do NOT calculate
anything yourself. Your only job is to decide which ONE tool should be
called next to help answer the user's question, based on the dataset
profile provided.

Respond with ONLY a JSON object, no other text, no markdown formatting,
in exactly this shape:

{"tool": "<tool_name>", "arguments": {<arguments as needed>}}

Example:
{"tool": "create_chart", "arguments": {"chart_type": "bar", "x_column": "product", "y_column": "revenue"}}
"""


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from the LLM's response text, even if the model
    wrapped it in markdown code fences or added stray text around it.
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Found a JSON-like block but couldn't parse it: {match.group(0)}") from e

    raise ValueError(f"No JSON object found in LLM response: {text}")


def plan_next_action(user_question: str, dataset_profile: dict) -> dict:
    """
    Ask the LLM which tool to call next, given a user's question and the
    dataset's profile (from profile_dataset()).

    Returns
    -------
    dict
        Shape: {"tool": str, "arguments": dict}
    """
    prompt = f"""{PLANNER_INSTRUCTIONS}

{TOOLS_DESCRIPTION}

Dataset profile:
{json.dumps(dataset_profile, indent=2)}

User question: "{user_question}"

Which tool should be called next? Respond with ONLY the JSON object.
"""

    reply = ask_llm(prompt)
    decision = _extract_json(reply)

    if "tool" not in decision:
        raise ValueError(f"LLM response missing 'tool' key: {decision}")

    decision.setdefault("arguments", {})

    return decision


def explain_result(user_question: str, tool_name: str, tool_result) -> str:
    """
    Ask the LLM to explain a tool's raw result in plain, natural language,
    directly answering the user's original question.

    The LLM does NOT recalculate anything here — it only reasons over
    numbers/data we already computed in Python.
    """
    prompt = f"""You are a data analyst assistant. A user asked a question,
and a tool already computed the exact factual result below. Your job is
to answer the user's question in clear, natural language using ONLY the
data provided. Do not invent numbers that aren't in the result. Be concise
but specific (mention actual numbers/values from the result).

User question: "{user_question}"

Tool used: {tool_name}
Tool result:
{json.dumps(tool_result, indent=2, default=str)}

Write a direct answer to the user's question, as if you're talking to them:
"""

    return ask_llm(prompt)