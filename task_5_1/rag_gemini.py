import os
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

from prompt_builder import build_prompt

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="ai_job_report"
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

question = input("Enter your question: ")

query_embedding = model.encode([question])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

retrieved_chunks = results["documents"][0]
retrieved_ids = results["ids"][0]

prompt = build_prompt(
    retrieved_chunks,
    question
)

response = gemini_client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response.text)

print("\nSOURCES:")

for i, chunk_id in enumerate(retrieved_ids, start=1):
    print(f"[{i}] {chunk_id}")