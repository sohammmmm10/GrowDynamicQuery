import streamlit as st
import pandas as pd
import os
import json
import re
import requests
from pandasql import sqldf
import plotly.express as px
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import snowflake.connector
from sentence_transformers import SentenceTransformer
import faiss

# ------------------ CONFIG ------------------
st.set_page_config(page_title="📊 Grow Chatbot", layout="wide")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.image("data:image/jpeg;base64,<your_base64_logo>", width=120)
    st.title("👤 User Panel")
    role = st.selectbox("Role", ["MR", "RSM", "ZSM", "Admin"])
    employee_id = st.text_input("Employee ID (Masked)")

    # Mask PII input display
    if employee_id:
        masked_id = employee_id[:2] + "*" * (len(employee_id) - 4) + employee_id[-2:]
        st.caption(f"🔒 Using masked Employee ID: {masked_id}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state["chat_history"] = []
        if os.path.exists("chat_history.pkl"):
            os.remove("chat_history.pkl")
        st.success("✅ Chat history cleared.")
        st.rerun()

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align:center; color:#4a7ebb;'>Grow Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Ask questions based only on the uploaded CSV data</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ SNOWFLAKE DATA LOAD ------------------
@st.cache_data
def load_data_from_snowflake():
    conn = snowflake.connector.connect(
        user="****",  # Masked
        password="****",  # Masked
        account="****",  # Masked
        warehouse="PROD_BI_XS_WH",
        database="PROD_DB",
        schema="DATAMART_CRM"
    )
    query = "SELECT * FROM PROD_DB.DATAMART_CRM.VW_GROW_EXPORT"
    cursor = conn.cursor()
    cursor.execute(query)
    df = pd.DataFrame.from_records(iter(cursor), columns=[col[0] for col in cursor.description])
    cursor.close()
    conn.close()

    schema = [{'column': col, 'dtype': str(df[col].dtype)} for col in df.columns]
    return df, schema

uploaded_df, schema = load_data_from_snowflake()

# ------------------ ROLE FILTER ------------------
def filter_data(df):
    try:
        if role == "MR":
            return df[df["EMPLOYEECODE"] == employee_id]
        elif role == "RSM":
            return df[df["RM_EMPLOYEE_ID"] == employee_id]
        elif role == "ZSM":
            return df[df["ZM_EMPLOYEE_ID"] == employee_id]
    except KeyError:
        st.error("❌ Employee ID column not found.")
        return df
    return df

uploaded_df = filter_data(uploaded_df)

# ------------------ CHAT HISTORY ------------------
if os.path.exists("chat_history.pkl") and "chat_history" not in st.session_state:
    with open("chat_history.pkl", "rb") as f:
        st.session_state["chat_history"] = pickle.load(f)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ------------------ OLLAMA LOCAL LLM FUNCTION ------------------
def run_llm(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",  # Change to your pulled Ollama model
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"❌ Ollama Error: {str(e)}"

# ------------------ SQL PROMPT PROCESSOR ------------------
def process_user_question(user_question):
    prompt = f"""
You are a data analyst restricted to ONLY use this schema:
{json.dumps(schema)}

User Question: "{user_question}"

RULES:
1. You MUST use only the table name: uploaded_df
2. DO NOT use placeholders like "your_table_name" or "table_name"
3. If the user asks about columns NOT present in the schema, respond:
{{"query": "", "explanation": "❌ One or more columns mentioned are not available in the dataset."}}
4. If valid, respond strictly in JSON:
{{"query": "<valid SQLite query using uploaded_df>", "explanation": "<brief explanation>"}}
"""
    raw_output = run_llm(prompt)
    try:
        match = re.search(r'{.*}', raw_output.strip(), re.DOTALL)
        if not match:
            return {"question": user_question, "query": "", "explanation": "", "error": "⚠️ LLM did not return valid JSON."}
        parsed = json.loads(match.group())
        query = parsed.get("query", "").replace("your_table_name", "uploaded_df").replace("table_name", "uploaded_df")
        explanation = parsed.get("explanation", "")

        if not query:
            return {"question": user_question, "query": "", "explanation": explanation, "result_df": None}

        result_df = sqldf(query, {"uploaded_df": uploaded_df})
        if result_df.empty:
            return {"question": user_question, "query": query, "explanation": explanation + " ⚠️ No matching results found.", "result_df": result_df}

        return {"question": user_question, "query": query, "explanation": explanation, "result_df": result_df}

    except Exception as e:
        return {"question": user_question, "error": f"❌ Error processing: {str(e)}"}

# ------------------ TABS ------------------
tabs = st.tabs(["💬 Chatbot", "🔮 RCPA Prediction", "📣 Smart Alerts", "🌍 Doctor Map", "📊 Dashboard"])

# ------------------ CHATBOT TAB ------------------
with tabs[0]:
    st.subheader("💬 Ask Your Question")
    with st.form("chat_form", clear_on_submit=True):
        new_question = st.text_input("Ask a question:")
        submitted = st.form_submit_button("Submit")
    if submitted and new_question:
        result = process_user_question(new_question)
        st.session_state["chat_history"].append(result)
        with open("chat_history.pkl", "wb") as f:
            pickle.dump(st.session_state["chat_history"], f)
        st.rerun()

    st.markdown("---")
    st.subheader("📚 Chat History")
    for i, chat in enumerate(st.session_state["chat_history"]):
        st.markdown(f"**🧑 You:** {chat['question']}")
        if "error" in chat:
            st.error(chat["error"])
        elif not chat["query"]:
            st.warning(chat["explanation"])
        else:
            st.markdown(f"**🤖 LLM Query:**\n```sql\n{chat['query']}\n```")
            with st.expander("🧠 Explanation"):
                st.markdown(chat["explanation"])
            if chat["result_df"] is not None and not chat["result_df"].empty:
                st.dataframe(chat["result_df"].head(), use_container_width=True)
                with st.expander("📊 Visualize"):
                    cols = chat["result_df"].columns.tolist()
                    if len(cols) >= 2:
                        x = st.selectbox("X-axis", cols, key=f"x_{i}")
                        y = st.selectbox("Y-axis", cols, key=f"y_{i}")
                        chart = st.selectbox("Chart", ["Bar", "Line", "Pie"], key=f"chart_{i}")
                        if chart == "Bar":
                            st.plotly_chart(px.bar(chat["result_df"], x=x, y=y), use_container_width=True)
                        elif chart == "Line":
                            st.plotly_chart(px.line(chat["result_df"], x=x, y=y), use_container_width=True)
                        elif chart == "Pie":
                            st.plotly_chart(px.pie(chat["result_df"], names=x, values=y), use_container_width=True)

# ------------------ RCPA Prediction ------------------
with tabs[1]:
    st.subheader("🔮 Predict Next Month's RCPA")
    doctor_list = uploaded_df["DOCTORNAME"].dropna().unique().tolist()
    selected_doctor = st.selectbox("Select Doctor", doctor_list)
    rcpa_cols = [col for col in uploaded_df.columns if re.match(r'[A-Z]{3}\d{2}RCPA', col)]

    if st.button("Predict Sales"):
        try:
            df_doctor = uploaded_df[uploaded_df["DOCTORNAME"] == selected_doctor]
            rcpa_series = df_doctor[rcpa_cols].mean().dropna().reset_index()
            rcpa_series.columns = ["Month", "RCPA"]
            rcpa_series["Month_Num"] = range(1, len(rcpa_series)+1)

            model = LinearRegression()
            model.fit(rcpa_series[["Month_Num"]], rcpa_series["RCPA"])
            predicted_rcpa = model.predict([[len(rcpa_series)+1]])[0]

            st.success(f"📈 Predicted RCPA for **{selected_doctor}**: **{round(predicted_rcpa)}**")
            fig = px.line(rcpa_series, x="Month", y="RCPA", title="Historical RCPA Trend")
            fig.add_scatter(x=["Next"], y=[predicted_rcpa], mode='markers+text', name="Prediction", text=["Predicted"], textposition="top center")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Prediction Error: {str(e)}")

# ------------------ SMART ALERTS ------------------
with tabs[2]:
    st.subheader("📣 Doctors with Drop in RCPA")
    if "DEC24RCPA" in uploaded_df.columns and "JAN25RCPA" in uploaded_df.columns:
        alert_df = uploaded_df[uploaded_df["JAN25RCPA"] < uploaded_df["DEC24RCPA"]]
        if not alert_df.empty:
            st.dataframe(alert_df[["DOCTORNAME", "DEC24RCPA", "JAN25RCPA"]])
        else:
            st.success("✅ No RCPA drop detected.")

# ------------------ DOCTOR MAP ------------------
with tabs[3]:
    st.subheader("🌍 Doctor Location Map")
    try:
        import folium
        from streamlit_folium import st_folium
        if "LOCATION" in uploaded_df.columns:
            m = folium.Map(location=[20.59, 78.96], zoom_start=5)
            for _, row in uploaded_df.dropna(subset=["LOCATION"]).iterrows():
                if isinstance(row["LOCATION"], str) and "," in row["LOCATION"]:
                    lat, lon = map(float, row["LOCATION"].split(","))
                    folium.Marker([lat, lon], popup=row.get("DOCTORNAME", "Doctor")).add_to(m)
            st_folium(m, width=700)
        else:
            st.warning("📍 'LOCATION' column not found.")
    except ImportError:
        st.warning("🛠️ Install folium and streamlit-folium for map support")

# ------------------ DASHBOARD ------------------
with tabs[4]:
    st.subheader("📊 Role-Based Insights")
    if role == "MR":
        st.markdown("#### 👨‍⚕️ MR Dashboard")
        rcpa_cols = [col for col in uploaded_df.columns if re.match(r'[A-Z]{3}\d{2}RCPA', col)]
        uploaded_df["TOTAL_RCPA"] = uploaded_df[rcpa_cols].sum(axis=1)
        top_docs = uploaded_df.groupby("DOCTORNAME")["TOTAL_RCPA"].sum().reset_index().sort_values(by="TOTAL_RCPA", ascending=False).head(5)
        st.dataframe(top_docs)

    elif role == "RSM":
        st.markdown("#### 🗺️ RSM Dashboard")
        if "PATCH" in uploaded_df.columns and "JAN25RCPA" in uploaded_df.columns:
            perf = uploaded_df.groupby("PATCH")["JAN25RCPA"].sum().reset_index()
            st.plotly_chart(px.bar(perf, x="PATCH", y="JAN25RCPA"), use_container_width=True)

    elif role == "ZSM":
        st.markdown("#### 📈 ZSM Dashboard")
        if "ZONE" in uploaded_df.columns:
            zone_perf = uploaded_df.groupby("ZONE")["JAN25RCPA"].sum().reset_index()
            st.plotly_chart(px.bar(zone_perf, x="ZONE", y="JAN25RCPA"), use_container_width=True)

# ------------------ MEMORY & FEEDBACK ------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
INDEX_FILE = "query_index.faiss"
MEMORY_FILE = "query_memory.json"

if os.path.exists(INDEX_FILE) and os.path.exists(MEMORY_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(MEMORY_FILE, "r") as f:
        stored_queries = json.load(f)
else:
    index = faiss.IndexFlatL2(384)
    stored_queries = []

def store_successful_query(query, explanation):
    embedding = embedder.encode([query])
    index.add(embedding)
    stored_queries.append({"query": query, "explanation": explanation})
    faiss.write_index(index, INDEX_FILE)
    with open(MEMORY_FILE, "w") as f:
        json.dump(stored_queries, f)

def get_similar_queries(question, top_k=3):
    if len(stored_queries) == 0:
        return []
    embedding = embedder.encode([question])
    D, I = index.search(embedding, top_k)
    return [stored_queries[i] for i in I[0] if i < len(stored_queries)]



if st.session_state["chat_history"]:
    last_chat = st.session_state["chat_history"][-1]
    if last_chat.get("query") and last_chat.get("explanation") and not last_chat.get("error"):
        store_successful_query(last_chat["question"], last_chat["explanation"])

with st.expander("💡 Similar Past Questions"):
    if st.session_state["chat_history"]:
        last_q = st.session_state["chat_history"][-1]["question"]
        similar = get_similar_queries(last_q)
        if similar:
            for i, item in enumerate(similar):
                st.markdown(f"**Suggestion {i+1}:** {item['query']}")
                st.caption(f"_Explanation_: {item['explanation']}")
        else:
            st.write("No similar past questions found.")

with st.expander("📣 Provide Feedback to Improve Responses"):
    feedback = st.text_area("What should the bot improve?")
    if st.button("Submit Feedback"):
        with open("feedback_log.txt", "a") as f:
            f.write(f"\nQuestion: {last_q}\nFeedback: {feedback}\n---\n")
        st.success("✅ Feedback submitted successfully.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 All Rights Reserved | © 2025 Grow Chatbot</p>", unsafe_allow_html=True)
