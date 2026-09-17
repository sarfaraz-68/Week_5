def build_prompt(chunks, chunk_ids, question):

    context = ""

    for i, chunk in enumerate(chunks):
        context += f"[{chunk_ids[i]}] {chunk}\n\n"

    prompt = f"""Answer the question using ONLY the context below.

If the answer is not present in the context, say:
"I don't have that information in the documents."

Do not use outside knowledge.
Do not invent or guess.
Cite the exact source chunk IDs used in your answer.

Context:

{context}

Question: {question}

Answer:"""

    return prompt