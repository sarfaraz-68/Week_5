from sentence_transformers import SentenceTransformer
from chunker import extract_text_from_pdf, chunk_text


model = SentenceTransformer("all-MiniLM-L6-v2")

pdf_path = "../documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = model.encode(chunks)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding:")
print(embeddings[0])

print("\nEmbedding dimensions:", len(embeddings[0]))