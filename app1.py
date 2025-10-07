# sales_chatbot_dynamic_schema_gemini.py
import streamlit as st
import pandas as pd
import pyodbc
import os
import re
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------
# 0. Load Gemini API key
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please create a .env file with GEMINI_API_KEY.")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# ---------------------------
# 1. Database Connection
# ---------------------------
def get_sql_connection():
    conn = pyodbc.connect(
        "Driver={SQL Server};"
        "Server=BS-praghottam\\SQLEXPRESS;"
        "Database=AdventureWorks2022;"
        "Trusted_Connection=yes;"
    )
    return conn

# ---------------------------
# 2. Fetch database metadata dynamically
# ---------------------------
@st.cache_data
def get_db_metadata():
    conn = get_sql_connection()
    query = """
    SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def generate_schema_prompt():
    df = get_db_metadata()
    prompt_lines = []
    for (schema, table), group in df.groupby(['TABLE_SCHEMA', 'TABLE_NAME']):
        columns = group['COLUMN_NAME'].tolist()
        prompt_lines.append(f"{schema}.{table} ({', '.join(columns)})")
    return "Use the following schemas and tables with columns:\n" + "\n".join(prompt_lines)

# ---------------------------
# 3. Run SQL Query
# ---------------------------
def run_query(query):
    conn = get_sql_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------------------------
# 4. Create Chatbot with Gemini
# ---------------------------
def create_chatbot():
    dynamic_schema_prompt = generate_schema_prompt()

    system_prompt = f"""
You are a SQL and Business Intelligence expert for the AdventureWorks2022 database.
{dynamic_schema_prompt}

Rules:
1. Use exact column names and proper joins to avoid SQL errors.
2. For customer names, join Sales.Customer → Person.Person using PersonID → BusinessEntityID.
3. For store names, join Sales.Customer → Sales.Store using StoreID → BusinessEntityID (use LEFT JOIN if needed).
4. Use TOP N correctly (after SELECT) in T-SQL.
5. Return ONLY valid SQL if user requests SQL.
6. For explanations or summaries, provide concise text.
7. For trends or performance, return both SQL and a short textual analysis.
8. Do not include markdown fences or extra text in SQL output.
"""
    user_prompt = "User Question: {question}"
    full_prompt = system_prompt + "\n" + user_prompt
    prompt_template = PromptTemplate(template=full_prompt, input_variables=["question"])

    # Correct Gemini LLM initialization
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        model_kwargs={"api_key": GEMINI_API_KEY},
        temperature=0
    )
    return LLMChain(llm=llm, prompt=prompt_template)

# ---------------------------
# 5. Extract SQL from LLM response
# ---------------------------
def extract_sql(raw_text):
    raw_text = re.sub(r"```(sql)?", "", raw_text, flags=re.IGNORECASE)
    lines = raw_text.splitlines()
    sql_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not started and re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", stripped, re.IGNORECASE):
            started = True
        if started:
            if re.match(r"^(This query|Here is|Explanation|Note|Rules|Use)", stripped, re.IGNORECASE):
                break
            sql_lines.append(line)
    return "\n".join(sql_lines).strip()

# ---------------------------
# 6. Streamlit UI
# ---------------------------
st.set_page_config(page_title="AdventureWorks Sales Chatbot", layout="wide")

st.markdown("""
<style>
body, .stApp { background-color: #343541; color: #ffffff; font-family: 'Helvetica', sans-serif'; }
.css-18e3th9 h1 { color: #ffffff; text-align: center; font-weight: bold; }
.user-msg { background-color: #444654; color: #ffffff; padding: 12px 15px; border-radius: 15px; margin: 10px 0; max-width: 80%; width: fit-content; }
.bot-msg { background-color: #10a37f; color: #ffffff; padding: 12px 15px; border-radius: 15px; margin: 10px 0; max-width: 80%; width: fit-content; }
.stTextInput>div>div>input { background-color: #202123; color: #ffffff; border-radius: 10px; }
.stDataFrame>div>div>div>div { background-color: #343541 !important; color: #ffffff !important; }
.stCodeBlock pre { background-color: #202123 !important; color: #f5f5f5 !important; font-family: 'Courier New', monospace; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("💬 AdventureWorks Sales Q&A Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa = create_chatbot()

# Sidebar chat history
st.sidebar.header("Chat History")
selected_index = st.sidebar.radio(
    "Select a previous conversation:",
    options=range(len(st.session_state.chat_history)),
    format_func=lambda i: f"Q{i+1}: {st.session_state.chat_history[i]['question']}"
) if st.session_state.chat_history else None

if selected_index is not None:
    entry = st.session_state.chat_history[selected_index]
    st.sidebar.markdown("**Question:**")
    st.sidebar.write(entry['question'])
    st.sidebar.markdown("**SQL/Answer:**")
    if entry.get('sql'):
        st.sidebar.code(entry['sql'], language="sql")
    if entry.get('summary'):
        st.sidebar.write(entry['summary'])
    if entry.get('df') is not None:
        st.sidebar.dataframe(entry['df'])

# Main chat input
user_input = st.text_input("Ask a question about AdventureWorks data:")

if user_input:
    response_dict = qa.invoke({"question": user_input})
    raw_response = response_dict.get("text") if isinstance(response_dict, dict) else str(response_dict)

    st.markdown(f'<div class="user-msg">{user_input}</div>', unsafe_allow_html=True)

    sql_query = extract_sql(raw_response)
    sql_result = None
    summary_text = None

    if sql_query:
        st.markdown(f'<div class="bot-msg">Generated SQL:</div>', unsafe_allow_html=True)
        st.code(sql_query, language="sql")

        try:
            sql_result = run_query(sql_query)
            if sql_result.empty:
                st.warning("Query executed successfully but returned 0 rows.")
            else:
                st.markdown(f'<div class="bot-msg">SQL Result:</div>', unsafe_allow_html=True)
                st.dataframe(sql_result)

                # Use Gemini LLM for summary as well
                summary_prompt = "Summarize this table in a few sentences:\n" + sql_result.head(10).to_string()
                summary_chain = LLMChain(
                    llm=ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        model_kwargs={"api_key": GEMINI_API_KEY},
                        temperature=0
                    ),
                    prompt=PromptTemplate(template="{question}", input_variables=["question"])
                )
                summary_response = summary_chain.invoke({"question": summary_prompt})
                summary_text = summary_response.get("text") if isinstance(summary_response, dict) else str(summary_response)
                st.markdown(f'<div class="bot-msg">Summary:</div>{summary_text}', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error running SQL: {e}")
    else:
        st.markdown(f'<div class="bot-msg">{raw_response}</div>', unsafe_allow_html=True)

    st.session_state.chat_history.append({
        'question': user_input,
        'sql': sql_query if sql_query else None,
        'summary': summary_text,
        'df': sql_result if sql_result is not None else None
    })

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for i, entry in enumerate(st.session_state.chat_history):
        st.markdown(f'<div class="user-msg">Q{i+1}: {entry["question"]}</div>', unsafe_allow_html=True)
        if entry.get('sql'):
            st.markdown(f'<div class="bot-msg">Generated SQL:</div>', unsafe_allow_html=True)
            st.code(entry["sql"], language="sql")
        if entry.get('summary'):
            st.markdown(f'<div class="bot-msg">Summary:</div>', unsafe_allow_html=True)
            st.write(entry["summary"])
        if entry.get('df') is not None:
            st.markdown(f'<div class="bot-msg">Result DataFrame:</div>', unsafe_allow_html=True)
            st.dataframe(entry['df'])
        st.write("---")
