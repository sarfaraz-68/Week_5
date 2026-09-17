import chromadb
from sentence_transformers import SentenceTransformer

from prompt_builder import build_prompt

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="ai_job_report"
)

question = "What are the top requested skills in AI jobs?"

query_embedding = model.encode([question])

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=3
)

retrieved_chunks = results["documents"][0]
retrieved_ids = results["ids"][0]
retrieved_distances = results["distances"][0]

prompt = build_prompt(
    retrieved_chunks,
    retrieved_ids,
    question
) 

print("\nQUESTION:")
print(question)

print("\nRETRIEVED SOURCES:")

for i in range(len(retrieved_chunks)):
    print(
        f"[{i + 1}] {retrieved_ids[i]} "
        f"(distance: {retrieved_distances[i]:.4f})"
    )

print("\nGROUNDED PROMPT:")
print(prompt)  