# CODE WHICH SAVES FAISS VECTORS LOCALLY

import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

# Load environment variables
load_dotenv()

# Path to your CSV file
clinical_csv_path = "ctg-studies (1).csv"

# Output FAISS vector DB path
save_path = "faiss_index"

def preprocess_data(csv_path):
    try:
        df = pd.read_csv(csv_path)

        def combine_fields(row):
            return (
                f"Study Title: {row.get('Study Title', '')}\n"
                f"Conditions: {row.get('Conditions', '')}\n"
                f"Primary Outcome Measures: {row.get('Primary Outcome Measures', '')}\n"
                f"Sex: {row.get('Sex', '')}\n"
                f"Age: {row.get('Age', '')}\n"
                f"Study Type: {row.get('Study Type', '')}\n"
                f"Sponsor: {row.get('Sponsor', '')}\n"
                f"Locations: {row.get('Locations', '')}"
            )

        df['combined_text'] = df.apply(combine_fields, axis=1)

        documents = [
            Document(page_content=text, metadata=row.to_dict())
            for text, (_, row) in zip(df['combined_text'], df.iterrows())
        ]
        return documents
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def build_and_save_vector_store(docs, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    
    if os.path.exists(save_path):
        print(f"⚠️ Vector store already exists at {save_path}. Skipping re-building.")
        return

    start_time = time.time()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)
    elapsed_time = round(time.time() - start_time, 2)
    
    print(f"✅ Vector store saved to: {save_path}")
    print(f"⏱ Time taken for vector store creation: {elapsed_time} seconds")

if __name__ == "__main__":
    docs = preprocess_data(clinical_csv_path)
    if docs:
        build_and_save_vector_store(docs, save_path)
