"""
planner.py

Step 34 of the AI Data Analyst Agent project.

Improved chart-type selection guidance: explicit rules and examples for
when to pick scatter, pie, and heatmap vs bar/line/histogram - a small
model doesn't reliably infer this from vague wording alone, so the
instructions now spell it out directly. A deterministic code-level
guardrail in app.py backs this up further.

Conversation history keeps more TURNS (14) by truncating each message
instead of dropping whole exchanges, preserving memory of earlier
context across a longer conversation.
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
     - chart_type: one of "bar", "line", "histogram", "scatter", "pie", "heatmap"
     - x_column: name of the column for the x-axis (not needed for heatmap)
     - y_column: name of the column for the y-axis (required for bar/line/scatter/pie, omit for histogram and heatmap)
   - Use "heatmap" when the user asks about correlation, relationships between
     numeric variables, or "how strongly X relates to Y" across many variables at once.
   - Use "scatter" for relationship between two specific numeric columns.
   - Use "pie" for share/percentage breakdown of a category.

4. chat
   - Purpose: respond to general conversation, greetings, or questions unrelated to the dataset.
   - Arguments: none.
"""

PLANNER_INSTRUCTIONS = """
You are a planning module for a data analysis agent. You do NOT calculate
anything yourself. Your only job is to decide which ONE tool should be
called next to help answer the user's question, based on the dataset
profile and the conversation so far.

IMPORTANT RULES:
- ANY question mentioning "the data", "this dataset", "that data", "it"
  (referring to the dataset), or asking what the data shows/represents/contains
  is ALWAYS a data question. Use profile_dataset (for structure: columns,
  types, missing values) or analyze_dataset (for statistics: averages, most
  common values, ranges) - never "chat" for these.
- Only use "chat" for clear non-data messages: greetings ("hi", "who are you"),
  thanks, or questions with absolutely nothing to do with the dataset.
- Only use create_chart if the user explicitly asks to see/show/plot/visualize
  something. Never pick create_chart for vague questions like "what do you see".
- Use the conversation history to understand follow-up questions. For example,
  if the user previously asked about "revenue by product" and now says
  "what about by region instead", understand they still want revenue, just
  grouped differently.

Chart type selection is important - match the request to the RIGHT chart_type:
- "chart"/"graph" with one category and one number, or "compare X across Y" -> bar
- "over time", "trend", "by date/month/year" -> line
- "distribution of X", "spread of X" -> histogram
- "X vs Y", "relationship between X and Y", "scatter" -> scatter (x_column and y_column both numerical)
- "percentage", "share", "proportion", "breakdown of X" -> pie (x_column categorical, y_column numerical)
- "correlation", "how are X, Y, Z related", "correlation matrix/heatmap", "relationships between all the numbers" -> heatmap (no x_column/y_column needed)

Examples:
"what does that data represent" -> analyze_dataset
"what do you see in this dataset" -> analyze_dataset
"are there missing values" -> profile_dataset
"show me a chart of revenue by product" -> create_chart, chart_type=bar, x_column=product, y_column=revenue
"revenue trend over time" -> create_chart, chart_type=line, x_column=date, y_column=revenue
"scatter plot of quantity vs price" -> create_chart, chart_type=scatter, x_column=quantity, y_column=price
"pie chart of revenue by product" -> create_chart, chart_type=pie, x_column=product, y_column=revenue
"what percentage of sales come from each region" -> create_chart, chart_type=pie, x_column=region, y_column=revenue
"show correlation between quantity, price, and revenue" -> create_chart, chart_type=heatmap
"how are the numeric columns related" -> create_chart, chart_type=heatmap
"hi" -> chat
"who are you" -> chat
"thanks!" -> chat

Respond with ONLY a JSON object, no other text, no markdown formatting,
in exactly this shape:

{"tool": "<tool_name>", "arguments": {<arguments as needed>}}
"""


def _format_history(history: list) -> str:
    """
    Turn a list of {"role": ..., "content": ...} messages into a simple
    readable transcript for the prompt. Keeps more TURNS than before by
    truncating each individual message instead of dropping whole
    exchanges - this preserves memory of earlier context (e.g. column
    names discussed several questions ago) while still limiting total
    prompt size.
    """
    if not history:
        return "(no previous conversation)"

    recent = history[-14:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 220:
            content = content[:217] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


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


def plan_next_action(user_question: str, dataset_profile: dict, history: list = None) -> dict:
    """
    Ask the LLM which tool to call next, given a user's question, the
    dataset's profile, and the recent conversation history.

    Returns
    -------
    dict
        Shape: {"tool": str, "arguments": dict}
    """
    history_text = _format_history(history or [])

    prompt = f"""{PLANNER_INSTRUCTIONS}

{TOOLS_DESCRIPTION}

Dataset profile:
{json.dumps(dataset_profile, indent=2)}

Conversation so far:
{history_text}

New user question: "{user_question}"

Which tool should be called next? Respond with ONLY the JSON object.
"""

    reply = ask_llm(prompt)
    decision = _extract_json(reply)

    if "tool" not in decision:
        raise ValueError(f"LLM response missing 'tool' key: {decision}")

    decision.setdefault("arguments", {})

    return decision


def explain_result(user_question: str, tool_name: str, tool_result, history: list = None) -> str:
    """
    Ask the LLM to explain a tool's raw result in plain, natural language,
    directly answering the user's question, taking the conversation history
    into account so follow-up questions make sense in context.
    """
    history_text = _format_history(history or [])

    prompt = f"""You are a data analyst assistant having an ongoing conversation
with a user about their dataset. A tool already computed the exact factual
result below for the user's latest question. Answer using ONLY the data
provided - do not invent numbers. Refer back to earlier parts of the
conversation when relevant (e.g. comparisons, "what about X instead").
Be concise but specific. Never include code, code blocks, or programming
syntax. Never use markdown tables (pipes | and dashes) or markdown bold
(**text**) - write in plain sentences and simple dashes for lists only.
This will be displayed as plain text, not rendered markdown.

Conversation so far:
{history_text}

New user question: "{user_question}"

Tool used: {tool_name}
Tool result:
{json.dumps(tool_result, indent=2, default=str)}

Write a direct answer to the user's question, as if you're talking to them:
"""

    return ask_llm(prompt)


def generate_insights(profile: dict, analysis: dict) -> str:
    """
    Look at the dataset's profile and full statistical analysis, and
    proactively give a natural, human-sounding first impression - the
    way a real analyst would open a conversation after glancing at a
    new file: first describing what it is, then reacting to what
    stands out.
    """
    prompt = f"""You are a data analyst who just opened a new dataset for the
first time. Based ONLY on the factual profile and statistics below, write a
short, natural, conversational reaction - like a colleague glancing at this
and talking out loud, not writing a report.

Structure it loosely like this (write it as flowing sentences, not labeled
sections or a table):

1. Briefly say how many rows and columns this has, and walk through what
   each column represents in plain words (e.g. "there's a date for when
   each order happened, a region and country for where, an item type for
   what was sold..."). Keep this part quick and natural, not a dry list.
2. Then shift into 2-3 genuine observations about what stands out - phrase
   them like real reactions ("I noticed...", "what's interesting is...",
   "one thing that stands out...").

Rules:
- Do not invent numbers not present below.
- Do not include code, markdown tables, or bold text (**text**).
- Do not use percentages unless they are directly calculable from counts given.
- Avoid a dense, listy, report-like tone. Write like a person, not a summary engine.
- Keep it to 5-7 sentences total.

Dataset profile:
{json.dumps(profile, indent=2)}

Full statistics:
{json.dumps(analysis, indent=2, default=str)}

Write the reaction now:
"""

    return ask_llm(prompt)