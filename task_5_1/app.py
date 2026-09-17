import os
import chromadb
import streamlit as st

from sentence_transformers import SentenceTransformer
from google import genai

from prompt_builder import build_prompt


model = SentenceTransformer("all-MiniLM-L6-v2")


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="ai_job_report"
)


gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


st.title("AI Job Market RAG Assistant")

st.write(
    "Ask a question about the AI Job Market report."
)


question = st.text_input(
    "Enter your question:"
)


if st.button("Ask") and question:

    query_embedding = model.encode([question])


    search_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3
    )


    retrieved_chunks = search_results["documents"][0]
    retrieved_ids = search_results["ids"][0]


    prompt = build_prompt(
        retrieved_chunks,
        retrieved_ids,
        question
    )


    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    st.subheader("Answer")

    st.write(response.text)


    st.subheader("Sources")

    for chunk_id in retrieved_ids:
        st.write(f"- {chunk_id}")