import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="ai_job_report"
)

query = "What are the top requested skills in AI jobs?"

query_embedding = model.encode([query])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

print("\nQUESTION:")
print(query)

print("\nTOP RESULTS:")

for i in range(3):
    print(f"\n--- RESULT {i + 1} ---")
    print("ID:", results["ids"][0][i])
    print("Distance:", results["distances"][0][i])
    print(results["documents"][0][i])