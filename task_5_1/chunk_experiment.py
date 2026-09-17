from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chunker import extract_text_from_pdf


def chunk_text(text, chunk_size, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_path = "../documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

query = "What are the top requested skills in AI jobs?"

query_embedding = model.encode([query])


for chunk_size in [300, 500, 800]:

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=50
    )

    embeddings = model.encode(chunks)

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = scores.argsort()[::-1][:3]

    print("\n" + "=" * 60)
    print(f"CHUNK SIZE: {chunk_size}")
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 60)

    for rank, index in enumerate(top_indices, start=1):

        print(f"\n--- RESULT {rank} ---")
        print("Chunk:", index + 1)
        print("Similarity:", round(scores[index], 4))
        print(chunks[index])