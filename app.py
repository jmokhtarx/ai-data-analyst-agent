"""
app.py

Step 34 of the AI Data Analyst Agent project.

Added a deterministic guardrail: if the planner picks "bar" chart_type
but both resolved columns are numerical (which makes no sense for a
bar chart - e.g. Quantity vs Price), the code overrides it to "scatter"
automatically, regardless of how the LLM interpreted the wording.

Everything else (multi-chat sidebar, renaming, slate/teal theme,
automatic insights on upload, cached analysis, professional report
generator, 6 chart types, data preview panel, deeper conversation
memory) is unchanged from previous steps.
"""

import json
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from tools.dataset_loader import load_dataset, profile_dataset, detect_column_type
from tools.analysis_tools import analyze_dataset
from tools.chart_tools import create_chart, find_column
from tools.planner import plan_next_action, explain_result, generate_insights
from tools.llm_client import ask_llm
from tools.report_generator import generate_pdf_report, generate_docx_report


st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data-analytics-inspired dark theme (slate + teal) ---
BG = "#0d1117"
SIDEBAR_BG = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CARD_BG = "#161b22"
BORDER = "#30363d"
ACCENT = "#2dd4bf"
USER_BUBBLE_BG = "#1a3a3a"
ACTIVE_CHAT_BG = "#1c2128"

st.markdown(f"""
<style>
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}
    .main .block-container {{
        padding-top: 2.5rem;
        max-width: 780px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT};
    }}
    h1, h2, h3, h4, p, span, div, label {{
        color: {TEXT};
    }}

    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] textarea {{
        background-color: {CARD_BG} !important;
        border-color: {BORDER} !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {TEXT} !important;
        border-radius: 16px !important;
        border: 1px solid {BORDER} !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border: 1px solid {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}
    [data-testid="stChatInput"] button {{
        background-color: {ACCENT} !important;
        border-radius: 10px !important;
    }}

    .dataset-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }}
    .dataset-card h4 {{
        margin: 0 0 6px 0;
        font-size: 0.9rem;
        color: {ACCENT};
    }}
    .dataset-card p {{
        margin: 2px 0;
        font-size: 0.82rem;
        color: {MUTED};
    }}
    .stButton button {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        text-align: left !important;
    }}
    .stButton button:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: {CARD_BG} !important;
        border: 1px dashed {BORDER} !important;
        border-radius: 12px;
    }}
    [data-testid="stFileUploaderDropzone"] * {{
        color: {MUTED} !important;
    }}
    [data-testid="stTextInput"] input {{
        background-color: {CARD_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border: 1px solid {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT} !important;
    }}
    [data-testid="stRadio"] label {{
        color: {TEXT} !important;
    }}
    [data-testid="stExpander"] {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {TEXT} !important;
    }}

    .chat-row {{
        display: flex;
        margin-bottom: 14px;
        width: 100%;
    }}
    .chat-row.user {{
        justify-content: flex-end;
    }}
    .chat-row.assistant {{
        justify-content: flex-start;
    }}
    .bubble {{
        max-width: 70%;
        padding: 10px 16px;
        border-radius: 16px;
        line-height: 1.5;
        font-size: 0.95rem;
        white-space: pre-wrap;
    }}
    .bubble.user {{
        background-color: {USER_BUBBLE_BG};
        color: {TEXT};
        border-bottom-right-radius: 4px;
    }}
    .bubble.assistant {{
        background-color: {CARD_BG};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-bottom-left-radius: 4px;
    }}
</style>
""", unsafe_allow_html=True)


def render_message(role: str, content: str):
    st.markdown(
        f'<div class="chat-row {role}"><div class="bubble {role}">{content}</div></div>',
        unsafe_allow_html=True,
    )


# --- Multi-chat session state setup ---
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat_id = None

if "renaming_chat_id" not in st.session_state:
    st.session_state.renaming_chat_id = None


def new_chat_id():
    return str(uuid.uuid4())[:8]


def create_new_chat():
    chat_id = new_chat_id()
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "df": None,
        "profile": None,
        "dataset_name": None,
        "analysis": None,
    }
    st.session_state.active_chat_id = chat_id
    st.session_state.renaming_chat_id = None


def switch_chat(chat_id):
    st.session_state.active_chat_id = chat_id
    st.session_state.renaming_chat_id = None


def delete_chat(chat_id):
    del st.session_state.chats[chat_id]
    if st.session_state.active_chat_id == chat_id:
        remaining = list(st.session_state.chats.keys())
        st.session_state.active_chat_id = remaining[0] if remaining else None
    st.session_state.renaming_chat_id = None


def start_rename(chat_id):
    st.session_state.renaming_chat_id = chat_id


def save_rename(chat_id, new_title):
    if new_title and new_title.strip():
        st.session_state.chats[chat_id]["title"] = new_title.strip()
    st.session_state.renaming_chat_id = None


def clear_active_chat():
    chat = st.session_state.chats[st.session_state.active_chat_id]
    chat["messages"] = chat["messages"][:1] if chat["messages"] else []


def reset_active_dataset():
    chat = st.session_state.chats[st.session_state.active_chat_id]
    chat["df"] = None
    chat["profile"] = None
    chat["dataset_name"] = None
    chat["analysis"] = None
    chat["messages"] = []
    chat["title"] = "New chat"


# Ensure there's always at least one chat
if not st.session_state.chats:
    create_new_chat()

active_id = st.session_state.active_chat_id
active_chat = st.session_state.chats[active_id]


# --- Sidebar ---
with st.sidebar:
    st.markdown("### AI Data Analyst")
    st.button("+ New chat", on_click=create_new_chat, use_container_width=True)
    st.divider()

    st.caption("CHATS")
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]

        if st.session_state.renaming_chat_id == chat_id:
            col_a, col_b = st.columns([5, 1])
            with col_a:
                new_title = st.text_input(
                    "rename",
                    value=chat["title"],
                    key=f"rename_input_{chat_id}",
                    label_visibility="collapsed",
                )
            with col_b:
                if st.button("OK", key=f"confirm_rename_{chat_id}", use_container_width=True):
                    save_rename(chat_id, new_title)
                    st.rerun()
        else:
            col_a, col_b, col_c = st.columns([4, 1, 1])
            with col_a:
                label = chat["title"] if chat["title"] else "New chat"
                is_active = chat_id == active_id
                prefix = "> " if is_active else ""
                if st.button(prefix + label, key=f"select_{chat_id}", use_container_width=True):
                    switch_chat(chat_id)
                    st.rerun()
            with col_b:
                if st.button("edit", key=f"rename_btn_{chat_id}", use_container_width=True):
                    start_rename(chat_id)
                    st.rerun()
            with col_c:
                if st.button("x", key=f"delete_{chat_id}", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()

    st.divider()

    if active_chat["df"] is None:
        st.caption("DATASET")
        uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            active_chat["df"] = load_dataset(tmp_path)
            active_chat["profile"] = profile_dataset(active_chat["df"])
            active_chat["dataset_name"] = uploaded_file.name
            if active_chat["title"] == "New chat":
                active_chat["title"] = uploaded_file.name
            active_chat["messages"].append({
                "role": "assistant",
                "content": f"Dataset loaded: {uploaded_file.name} "
                           f"({active_chat['profile']['rows']} rows, {active_chat['profile']['columns']} columns).",
            })

            with st.spinner("Looking for interesting patterns..."):
                active_chat["analysis"] = analyze_dataset(active_chat["df"])
                insights = generate_insights(active_chat["profile"], active_chat["analysis"])

            active_chat["messages"].append({
                "role": "assistant",
                "content": insights,
            })
            st.rerun()
    else:
        profile = active_chat["profile"]
        col_names = ", ".join(c["name"] for c in profile["columns_profile"])
        st.markdown(f"""
        <div class="dataset-card">
            <h4>{active_chat['dataset_name']}</h4>
            <p><b>{profile['rows']}</b> rows &nbsp;·&nbsp; <b>{profile['columns']}</b> columns</p>
            <p><b>{profile['duplicate_rows']}</b> duplicate rows</p>
            <p>{col_names}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.button("Clear chat", on_click=clear_active_chat, use_container_width=True)
        with c2:
            st.button("New dataset", on_click=reset_active_dataset, use_container_width=True)

        st.divider()
        st.caption("REPORT")
        report_format = st.radio(
            "Format", ["PDF", "Word (.docx)"], horizontal=True, label_visibility="collapsed", key="report_format"
        )
        if st.button("Generate report", use_container_width=True):
            with st.spinner("Building report..."):
                result_analysis = active_chat["analysis"]
                if report_format == "PDF":
                    report_path = generate_pdf_report(
                        active_chat["dataset_name"], profile, result_analysis, active_chat["messages"]
                    )
                    mime = "application/pdf"
                else:
                    report_path = generate_docx_report(
                        active_chat["dataset_name"], profile, result_analysis, active_chat["messages"]
                    )
                    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            with open(report_path, "rb") as f:
                st.download_button(
                    "Download report",
                    data=f.read(),
                    file_name=Path(report_path).name,
                    mime=mime,
                    use_container_width=True,
                )


# --- Main chat area ---
if active_chat["df"] is None:
    st.markdown("<h1 style='text-align:center; margin-top:3rem;'>What's in your data?</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:{MUTED};'>Upload a dataset from the sidebar to start chatting.</p>", unsafe_allow_html=True)
else:
    with st.expander(f"View data - {active_chat['dataset_name']} ({active_chat['profile']['rows']} rows x {active_chat['profile']['columns']} cols)"):
        st.dataframe(active_chat["df"], use_container_width=True, height=350)

    for message in active_chat["messages"]:
        render_message(message["role"], message["content"])
        if message.get("image"):
            st.image(message["image"])

    question = st.chat_input("Ask a question about your data...")

    if question:
        active_chat["messages"].append({"role": "user", "content": question})
        render_message("user", question)

        if active_chat["title"] in (active_chat["dataset_name"], "New chat"):
            short_title = question if len(question) <= 40 else question[:37] + "..."
            active_chat["title"] = short_title

        with st.spinner("Thinking..."):
            decision = plan_next_action(question, active_chat["profile"], active_chat["messages"])

        tool = decision["tool"]
        arguments = decision["arguments"]

        # Guardrail: if the planner picked "bar" but both resolved columns
        # are numerical, a bar chart doesn't make sense - override to scatter.
        if tool == "create_chart":
            requested_type = arguments.get("chart_type", "bar")
            x_col = find_column(active_chat["df"], arguments.get("x_column"))
            y_col = find_column(active_chat["df"], arguments.get("y_column"))
            if requested_type == "bar" and x_col and y_col:
                if (
                    detect_column_type(active_chat["df"][x_col]) == "numerical"
                    and detect_column_type(active_chat["df"][y_col]) == "numerical"
                ):
                    arguments["chart_type"] = "scatter"

        image_path = None

        with st.spinner("Working on your answer..."):
            if tool == "profile_dataset":
                reply_text = explain_result(question, tool, active_chat["profile"], active_chat["messages"])
            elif tool == "analyze_dataset":
                reply_text = explain_result(question, tool, active_chat["analysis"], active_chat["messages"])
            elif tool == "create_chart":
                try:
                    chart_info = create_chart(active_chat["df"], **arguments)
                    image_path = chart_info["path"]
                    if chart_info["chart_type"] == "heatmap":
                        reply_text = "Here's a correlation heatmap across the numeric columns:"
                    elif chart_info["y_column"] and chart_info["chart_type"] not in ("histogram",):
                        reply_text = f"Here's a {chart_info['chart_type']} chart showing {chart_info['y_column']} by {chart_info['x_column']}:"
                    else:
                        reply_text = f"Here's a {chart_info['chart_type']} chart of {chart_info['x_column']}:"
                except Exception:
                    reply_text = (
                        "I couldn't quite figure out what chart to make from that. "
                        "Could you tell me a bit more - like which column(s) you're interested in?"
                    )
            elif tool == "chat":
                reply_text = ask_llm(
                    f"You are a friendly data analyst assistant talking to a non-technical "
                    f"user in a chat interface. You're currently helping them analyze this dataset:\n"
                    f"{json.dumps(active_chat['profile'], indent=2)}\n\n"
                    f"Respond naturally to this message: \"{question}\"\n"
                    f"If the message is actually asking about the dataset, use the profile above "
                    f"to answer helpfully instead of asking which dataset they mean.\n\n"
                    f"IMPORTANT: Never include code, code blocks, or programming syntax (no Python, "
                    f"no library names like Matplotlib/Seaborn). Never use markdown tables or bold "
                    f"text (**text**) - write in plain sentences only, since this will be displayed "
                    f"as plain text. Keep it concise."
                )
            else:
                reply_text = f"The planner picked an unknown tool: {tool}"

        render_message("assistant", reply_text)
        if image_path:
            st.image(image_path)

        active_chat["messages"].append({
            "role": "assistant",
            "content": reply_text,
            "image": image_path,
        })