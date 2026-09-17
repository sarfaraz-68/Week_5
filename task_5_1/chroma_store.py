import chromadb
from sentence_transformers import SentenceTransformer

from chunker import extract_text_from_pdf, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_path = "../documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = model.encode(chunks)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="ai_job_report"
)

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)

print("Total chunks stored:", collection.count())