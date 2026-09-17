import os
import chromadb
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


print("RAG Application")
print("Type 'exit' to quit.")

while True:

    question = input("\nAsk a question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break


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


    print("\nANSWER:")
    print(response.text)


    print("\nRETRIEVED SOURCES:")

    for chunk_id in retrieved_ids:
        print(f"- {chunk_id}")