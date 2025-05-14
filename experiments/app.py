# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Load Data and Model ------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_csv("ctg-studies.csv")
    
    # Replace missing values
    df.fillna("Not available", inplace=True)
    
    # Combine fields
    def combine_fields(row):
        return f"{row['Study Title']} {row['Brief Summary']} {row['Conditions']} {row['Interventions']} {row['Primary Outcome Measures']}"
    
    df['combined_text'] = df.apply(combine_fields, axis=1)
    return df

@st.cache_data
def compute_embeddings(texts, _model):
    return _model.encode(texts, show_progress_bar=True)

# ------------------ Semantic Search ------------------

def semantic_search(query, df, embeddings, model, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="Clinical Trial Semantic Search", layout="wide")
st.title("🔍 Clinical Trial Semantic Search")
st.markdown("Enter a natural language query to find similar past clinical trials.")

# Load resources
model = load_model()
df = load_data()
embeddings = compute_embeddings(df['combined_text'].tolist(), _model=model)

# User input
query = st.text_input("Enter your query:", placeholder="e.g., lung cancer trial with immunotherapy")

top_k = st.slider("Number of similar trials to return", 1, 20, 5)

if st.button("Search") and query.strip() != "":
    with st.spinner("Searching..."):
        results = semantic_search(query, df, embeddings, model, top_k=top_k)
        st.success(f"Showing top {top_k} similar trials:")
        for _, row in results.iterrows():
            with st.expander(f"{row['Study Title']}"):
                st.markdown(f"**Summary**: {row['Brief Summary']}")
                st.markdown(f"**Conditions**: {row['Conditions']}")
                st.markdown(f"**Interventions**: {row['Interventions']}")
                st.markdown(f"**Outcomes**: {row['Primary Outcome Measures']}")
