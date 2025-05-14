import os
import streamlit as st
from dotenv import load_dotenv
import time
import requests

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page config
st.set_page_config(page_title="🧪 Clinical Trial Insight Assistant", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Semantic Insight Engine for Clinical Trials</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.markdown("### ℹ️ Instructions")
    st.write("Enter a natural language query like 'lung cancer trial with immunotherapy' to get top 10 semantically similar clinical studies.")
    st.markdown("Powered by **FAISS + HuggingFace + GROQ + LangChain**")
    st.markdown("---")
    st.caption("Developed by Team DataVerse")

# Main Form
with st.form("input_form"):
    st.markdown("### 📝 Enter Your Query")
    user_query = st.text_input("🔍 Example: 'lung cancer trial with immunotherapy'")
    submitted = st.form_submit_button("🔍 Search")

# Vector DB Loader
def load_vector_db():
    if "vectors" not in st.session_state:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.session_state.vectors = vector_store
        except Exception as e:
            st.error(f"❌ Failed to load vector DB: {e}")

# Summary via GROQ
def get_summary_from_groq(metadata_dict):
    try:
        prompt = f"""
        You are a helpful assistant. Summarize the following clinical trial information in simple English in 1 paragraph:
        {metadata_dict}
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Summary generation failed: {e}"

# Main logic
if submitted:
    load_vector_db()

    if "vectors" in st.session_state:
        user_input = user_query  # Direct query input

        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
        start = time.process_time()
        results = retriever.invoke(user_input)
        elapsed = round(time.process_time() - start, 2)

        st.success("✅ Retrieval Successful")
        st.markdown("## 🔍 Top 10 Matched Clinical Studies")

        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            score = getattr(doc, 'score', None)
            metadata["semantic_similarity_score"] = round(score, 4) if score else "N/A"
            metadata["summary"] = get_summary_from_groq(metadata)

            with st.expander(f"🔹 Study {i}: {metadata.get('Study Title', 'No Title')}"):
                st.json(metadata)

        st.info(f"⏱ Time Taken: `{elapsed} seconds`")
    else:
        st.error("🚫 Vector DB not loaded. Please try again.")
