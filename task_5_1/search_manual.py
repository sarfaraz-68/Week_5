from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import extract_text_from_pdf, chunk_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load document
pdf_path = "../documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)


# Create embeddings for document chunks
chunk_embeddings = model.encode(chunks)


# User's question
query = "What are the top requested skills in AI jobs?"


# Create embedding for the question
query_embedding = model.encode([query])


# Calculate similarity
scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)[0]


# Get indices of highest scores
top_indices = scores.argsort()[::-1][:3]


print("\nQUESTION:")
print(query)

print("\nTOP 3 RESULTS:")

for rank, index in enumerate(top_indices, start=1):

    print(f"\n--- RESULT {rank} ---")
    print("Chunk:", index + 1)
    print("Similarity:", round(scores[index], 4))
    print(chunks[index])