import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

from evaluation_questions import questions
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

question_number = int(
    input("Enter question number (1-10): ")
)

item = questions[question_number - 1]

question = item["question"]
expected = item["expected"]

print("\nQUESTION:")
print(question)

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

answer = response.text

refusal_text = "I don't have that information in the documents."

if expected == "refuse":
    passed = refusal_text.lower() in answer.lower()
else:
    has_citation = "[chunk_" in answer.lower()
    is_not_refusal = refusal_text.lower() not in answer.lower()
    passed = has_citation and is_not_refusal

result = {
    "question_number": question_number,
    "question": question,
    "expected": expected,
    "answer": answer,
    "sources": retrieved_ids,
    "pass": passed
}

print("\nANSWER:")
print(answer)

print("\nSOURCES:")
for chunk_id in retrieved_ids:
    print(chunk_id)

print("\nEXPECTED:")
print(expected)

print("\nPASS:")
print(passed)

with open(
    f"evaluation_result_{question_number}.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        result,
        file,
        indent=4,
        ensure_ascii=False
    )

print(
    f"\nSaved: evaluation_result_{question_number}.json"
)