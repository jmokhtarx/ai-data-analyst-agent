# AI Data Analyst Agent

A local-first AI agent that lets you upload a dataset (CSV or Excel) and chat with it in plain English. Instead of writing pandas code yourself, you ask questions — the agent decides which analysis or chart is needed, runs the real calculation in Python, and explains the result back to you in natural language.

Built incrementally, one tested step at a time, following the principle that **the LLM decides what to do, but Python always does the actual math.**

>  **Work in progress.** The core chat, dataset profiling, and automatic insights work end-to-end. The **charts** and **report export** features are still being polished and may behave inconsistently on some datasets.

---

## Gallery

![AI Data Analyst Agent interface](docs/1.png)
![Automatic insights on upload](docs/2.png)
![Multi-chat sidebar with dataset profile](docs/3.png)
![Downloadable PDF report](docs/4.png)
![Dataset profile card](docs/5.png)
![Chart example](docs/6.png)
![Q&A exchange](docs/7.png)


---

## How it works
User
|
v
Upload Dataset ---------------------------------------------+
| |
v |
Dataset Loader (tools/dataset_loader.py) |
| |
v |
Dataset Profiler (tools/dataset_loader.py) |
| |
v |
Analysis Planner (LLM) (tools/planner.py -> plan_next_action)
|
v
Tool Selection ---> profile_dataset | analyze_dataset | create_chart | chat
|
v
Python Analyst / Tools (tools/analysis_tools.py, tools/chart_tools.py)
|
v
Insight Analyst (LLM) (tools/planner.py -> explain_result / generate_insights)
|
v
Report Generator (tools/report_generator.py)
|
v
Final Answer / Downloadable Report (PDF or Word)

text

The LLM never invents numbers. It only:

- Decides which tool answers the question (the **Planner**)
- Turns the tool's real output into a plain-language answer (the **Explainer**)

Every number you see in the chat was computed by pandas, not guessed by the model.

---

## Features

- Upload CSV or Excel and get an instant structural profile (rows, columns, types, missing values, duplicates)
- **Automatic insights on upload** — the agent proactively points out what stands out before you ask anything
- Ask questions in plain English — stats, comparisons, *"what about X instead"* follow-ups, all understood in context of the ongoing conversation
- On-demand charts — bar, line, and histogram, with fuzzy column matching so typos or wrong casing still work
- Multi-chat sidebar — ChatGPT-style interface: create multiple chats, each with its own dataset and history, rename or delete any of them
- Downloadable reports — export a full session (dataset overview, column stats, every chart made, full Q&A) as a PDF or Word document
- Fast inference via Groq's free cloud API (open-weight models on purpose-built fast inference hardware)

---

## Tech stack

| Layer          | Tool                                                     |
| -------------- | -------------------------------------------------------- |
| Data handling  | Python, Pandas, NumPy                                    |
| Charts         | Matplotlib                                               |
| LLM inference  | Groq API (OpenAI-compatible), model: `openai/gpt-oss-20b` |
| Interface      | Streamlit                                                |
| Report export  | fpdf2 (PDF), python-docx (Word)                          |

No LangChain, no database, no deployment — kept intentionally simple for a local MVP, per the original project constraints.

---

## Project structure
AI_Data_Analyst/
|-- app.py # Streamlit UI + orchestration
|-- data/
| -- sales.csv # sample dataset |-- tools/ | |-- dataset_loader.py # load_dataset, profile_dataset, detect_column_type | |-- analysis_tools.py # analyze_dataset (descriptive stats) | |-- chart_tools.py # create_chart (bar/line/histogram, fuzzy matching) | |-- planner.py # plan_next_action, explain_result, generate_insights | |-- llm_client.py # ask_llm (Groq API connection) |-- report_generator.py # generate_pdf_report, generate_docx_report
|-- charts/ # generated chart images (auto-created)
|-- reports/ # generated report files (auto-created)
|-- docs/ # screenshots used in this README
`-- README.md

text

---

## Setup

### 1. Install dependencies

```bash
pip install pandas numpy matplotlib streamlit requests fpdf2 python-docx
2. Get a free Groq API key
Sign up at console.groq.com (no credit card needed)

Create an API key under API Keys

3. Set your API key (PowerShell, per terminal session)
powershell
$env:GROQ_API_KEY="gsk_your_actual_key_here"
4. Run the app
bash
python -m streamlit run app.py
The app opens automatically at http://localhost:8501.

The dataset used for testing
The agent was tested end-to-end on the Big Sales Data dataset from Kaggle:

Source: https://www.kaggle.com/datasets/pigment/big-sales-data

About: Time series analysis deals with time series based data to extract patterns for predictions and other characteristics of the data. It uses a model for forecasting future values in a small time frame based on previous observations. It is widely used for non-stationary data, such as economic data, weather data, stock prices, and retail sales forecasting.

The dataset was chosen because it exercises every part of the pipeline: mixed column types (dates, categoricals, numerics), non-trivial groupings (sales by product, by date), missing-value handling, and enough rows to make aggregation meaningful without being slow in a local MVP.

A trimmed sample (data/sales.csv) is included in the repo so the app can be demoed without downloading the full dataset.

User interface
The interface is a dark, analytics-tool-styled chat app (slate background, teal accent), split into a sidebar and a main chat column:

text
+---------------------------------------------------------------+
| SIDEBAR              |  MAIN CHAT AREA                        |
|-----------------------|-----------------------------------------|
| AI Data Analyst       |          What's in your data?          |
| [+ New chat]          |                                         |
|                        |  [assistant] Dataset loaded:           |
| CHATS                 |    big_sales.csv (16k rows, 6 cols)     |
| > sales_2026    [x]   |                                         |
|   test_sales    [x]   |  [assistant] So this looks like daily   |
|                        |    retail sales... I noticed the        |
| DATASET                |    Lightning Charging Cable dominates   |
| +----------------+    |    the product breakdown...             |
| | big_sales.csv  |    |                                         |
| | 16k rows,6 cols|    |                       [user] show me a  |
| | 93 duplicates  |    |                       chart of sales   |
| | Order ID,      |    |                       by product [you]  |
| | Product, Qty,  |    |                                         |
| | Price, Date,   |    |  [assistant] Here's a bar chart showing |
| | Address        |    |    total sales by product:              |
| +----------------+    |    [ chart image ]                      |
|                        |                                         |
| [Clear chat] [New set] |                                         |
|                        |  +-----------------------------------+ |
| REPORT                 |  | Ask a question about your data... | |
| (PDF) (Word)           |  +-----------------------------------+ |
| [Generate report]      |                                         |
+---------------------------------------------------------------+
Example questions to try
"What does this data represent?"

"How many rows and columns are in this dataset?"

"Are there any missing values?"

"What's the average price?"

"Show me a bar chart of Quantity Ordered by Product"

"What about the top 10 only?" (follow-up, uses conversation context)

"Who are you?" (routed to general chat, not treated as a data question)

Charts and report generation are still being polished — see Known limitations above.

Design principles
A few rules the project holds itself to:

The LLM plans, Python computes. Models are good at routing and phrasing, bad at arithmetic. Every number in a response comes out of pandas.

Tools are pure and testable. Each tool in tools/ takes data in and returns plain data out — no LLM calls inside them, no hidden state.

One tested step at a time. New capabilities are added incrementally, with a working demo after each step.

Local-first, minimal dependencies. No orchestration frameworks, no vector DBs, no cloud deployment. Just Python, Streamlit, and an API call.

Known limitations / next steps
Still in progress

Charts — the chart planner is being reworked. Requests that reference computed values (e.g. "revenue by product" when the dataset only stores quantity and price) can fail to route correctly. Asking for raw column names directly (e.g. "Quantity Ordered by Product") works reliably.

Reports — PDF/Word export works for basic sessions but is still being refined for edge cases (many charts, long Q&A histories, missing values).

Interface screenshots — the UI shown in the screenshots above is the current design and may change as these features are finished.

General limitations

Chats and uploaded data live only in memory for the current browser session — refreshing clears everything (no disk persistence yet)

No correlation or outlier detection yet

Tested on datasets up to ~20k rows; performance on larger files not yet verified

Planned / potential additions
Session persistence (SQLite or JSON on disk) so chats survive a refresh

Correlation matrix and outlier detection tools

Support for grouped/multi-series line charts

CSV export of a tool's raw output alongside the natural-language answer

"Explain this number" hover — click any figure in an answer to see the exact pandas call that produced it

Streaming responses from Groq for a snappier chat feel

