import os
import streamlit as st
from streamlit_option_menu import option_menu
import requests
import json
import lancedb
import time
import pandas as pd
import jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Streamlit Multi-Page App", layout="wide")

# Initialize progress bar
progress_bar = st.progress(0)

ingested_knowledgebases = []
DB_PATH = "lancedb_store"
os.makedirs(DB_PATH, exist_ok=True)
db = lancedb.connect(DB_PATH)
# Environment variables
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
INGESTION_ENDPOINT = os.getenv("INGESTION_ENDPOINT", "/upload/")
CHECK_STATUS_ENDPOINT = os.getenv("CHECK_STATUS_ENDPOINT", "/check_status/")
CONVERSATION_ENDPOINT = os.getenv("CONVERSATION_ENDPOINT", "/conversations/")
CONTINUE_CONVERSATION_ENDPOINT = os.getenv("CONTINUE_CONVERSATION_ENDPOINT", "/continue_conversations/")
PCA_ENDPOINT = os.getenv("PCA_ENDPOINT", "/conversations")
CONVERT_ENDPOINT = os.getenv("CONVERT_ENDPOINT")

def landing_page():
    st.title("Welcome to Our Streamlit App")
    st.write("This is the landing page of our multi-page application.")
    st.image("https://source.unsplash.com/800x400/?technology", caption="Landing Page Image")
    st.markdown("### Features:")
    st.markdown("- 🏠 Home Page")
    st.markdown("- 📄 Data Ingestion")
    st.markdown("- 🔍 Query Page")
    st.markdown("- 📊 Conversation Report")

def check_status(kb_id):
    """Check ingestion status from the endpoint."""
    check_status_url = f"{BASE_URL}{CHECK_STATUS_ENDPOINT}"
    try:
        response = requests.get(check_status_url, params={"knowledgebase_id": kb_id})
        data = response.json()
        return data.get("status", "error"), data.get("kb_names", [])
    except requests.exceptions.RequestException:
        return "error", []

def data_ingestion_page():
    st.title("Data Ingestion")
    if "ingested_knowledgebases" not in st.session_state:
       st.session_state.ingested_knowledgebases = []
    if "ingested_kb_id" not in st.session_state:
        st.session_state.ingested_kb_id = []
    if "selected_knowledgebase" not in st.session_state:
       st.session_state.selected_knowledgebase = None
    endpoint = f"{BASE_URL}{INGESTION_ENDPOINT}"
    knowledgebase_name = st.text_input("Knowledgebase Name")
    uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)
    
    if st.button("Submit"):
        if not knowledgebase_name or not uploaded_files:
            st.error("Please provide both knowledgebase name and files.")
            return
        
        status_text = st.empty()
        status_text.info("Data ingestion in progress...")
        
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(endpoint, data={"knowledgebase_name": knowledgebase_name}, files=files)
        response_data = response.json()
        kb_id = response_data.get("kb_id")
        while True:
            status, kb_names = check_status(kb_id=kb_id)
            if status == "success":
                status_text.success("✅ Data ingested successfully!")
                st.session_state.ingested_knowledgebases = kb_names
                st.session_state.ingested_kb_id.append(kb_id)
                break
            elif status == "error":
                status_text.error("❌ Failed to fetch status. Please check the API.")
                break
            time.sleep(2)

def query_page():
   
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = []
    if "conversation_id_store" not in st.session_state:
        st.session_state.conversation_id_store = []
    if "selected_conv_id" not in st.session_state:
        st.session_state.selected_conv_id = st.session_state.conversation_id[0] if st.session_state.conversation_id else ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_kb" not in st.session_state:
        st.session_state.selected_kb = st.session_state.ingested_knowledgebases[0] if st.session_state.ingested_knowledgebases else ""
    st.title("💬 AI Chatbot")
    st.write("Ask your question below:")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if st.session_state.ingested_knowledgebases:
        st.session_state.selected_kb = st.sidebar.selectbox(
            "Select a knowledge base:",
            st.session_state.ingested_knowledgebases,
            index=0
        )
    query = st.chat_input("Enter your query...")
    if query:
        knowledgebase_name = st.session_state.selected_kb
        result= requests.get(f"{BASE_URL}{CONVERT_ENDPOINT}",params={"kb_name":knowledgebase_name})
        kb_id = result.json().get('kb_id')
        if not st.session_state.selected_conv_id:
            response = requests.post(f"{BASE_URL}{CONVERSATION_ENDPOINT}", params={"query": query, "kb_id": kb_id})
            if response.status_code == 200:
                bot_reply = response.json().get("answer", "No response received.")
                new_conv_id = response.json().get("conversation_id")
                conv_list = response.json().get("conv_list")
                st.session_state.conversation_id_store = conv_list
                
            else:
                bot_reply = f"❌ Error in creating : {response.status_code}"
        else:

            response = requests.post(f"{BASE_URL}{CONTINUE_CONVERSATION_ENDPOINT}", params={"query": query, "kb_id": kb_id, "conversation_id":st.session_state.selected_conv_id})
            if response.status_code == 200:
                bot_reply = response.json().get("answer", "No response received.")
            else:
                bot_reply = f"❌ Error in continue: {response.status_code}"
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()
    
    if st.session_state.conversation_id_store:
        st.session_state.selected_conv_id = st.sidebar.selectbox(
            "Select ocnversation ID:",
            st.session_state.conversation_id_store,
            index=0
        )
    st.session_state.conversation_id = st.session_state.conversation_id_store
    
    if st.sidebar.button("New Chat"):
    # Reset state before rerun
        st.session_state.messages = []
        st.session_state.selected_conv_id = ""
        st.session_state.conversation_id = []
        st.rerun()

def conversation_report():
    st.title("Conversation Report")
    selected_ids = st.multiselect(
    "Select Conversations to Generate Report:",
    st.session_state.conversation_id_store,
    default=st.session_state.conversation_id_store  # Select all by default
)

# Function to fetch report for a single conversation ID
    def fetch_report(conversation_id):
        response = requests.post(f'{BASE_URL}{PCA_ENDPOINT}', params={"conversation_id": conversation_id})
        if response.status_code == 200:
            data = response.json()
            # data["conversation_id"] = conversation_id  # Ensure ID is included
            return data
        else:
            return {"conversation_id": conversation_id, "sentiment_score": "N/A", "knowledge_gap": [], "topic": [], "error": f"Error {response.status_code}"}

    # Button to generate report
    if st.button("Generate Report"):
        report_data = [fetch_report(cid) for cid in selected_ids]
        if report_data:
            # Display the report in a clean table format
            st.subheader("📋 Report Summary")
            i=0
            for data in report_data:
                st.write(f"### 🆔 Conversation ID:{st.session_state.conversation_id_store[i]}")
                i+=1

                # Color-coded sentiment score
                sentiment = data.get("sentiment_score", "N/A")
                if isinstance(sentiment, int):  
                    if sentiment > 5:
                        sentiment_text = f"🟢 **{sentiment} (Positive)**"
                    elif sentiment < 5:
                        sentiment_text = f"🔴 **{sentiment} (Negative)**"
                    else:
                        sentiment_text = f"🟡 **{sentiment} (Neutral)**"
                else:
                    sentiment_text = f"⚪ **N/A**"

                st.markdown(f"**Sentiment Score:** {sentiment_text}")

                # Display Knowledge Gap as bullet points
                knowledge_gap = data.get("knowledge_gap", [])
                if knowledge_gap:
                    st.markdown("**📚 Knowledge Gaps:**")
                    for item in knowledge_gap:
                        st.write(f"- {item}")
                else:
                    st.write("✅ No knowledge gaps detected.")

                # Display Topics as bullet points
                topic = data.get("topic", [])
                if topic:
                    st.markdown("**📌 Topics Discussed:**")
                    for item in topic:
                        st.write(f"- {item}")
                else:
                    st.write("No topics available.")

                st.divider()  # Adds a visual separator

            # Option to download report as JSON
            json_str = json.dumps(report_data, indent=4)
            st.download_button(
                label="📥 Download Report as JSON",
                data=json_str,
                file_name="conversation_report.json",
                mime="application/json"
            )
        else:
            st.warning("No data available for the selected conversation IDs.")

def main():
    with st.sidebar:
        page = option_menu("Navigation", ["Landing Page", "Data Ingestion", "Query Page", "Conversation Report"], 
                           icons=["house", "database", "search", "bar-chart"], menu_icon="menu")
    
    if page == "Landing Page":
        landing_page()
    elif page == "Data Ingestion":
        data_ingestion_page()
    elif page == "Query Page":
        query_page()
    elif page == "Conversation Report":
        conversation_report()

if __name__ == "__main__":
    main()
