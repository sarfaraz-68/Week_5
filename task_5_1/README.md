# Task 5.1 — RAG & Grounded Document Question Answering

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions from an AI Job Market technical report.

The system does not rely only on the LLM's general knowledge. Instead, it:

1. Loads the PDF document.
2. Splits the document into smaller chunks.
3. Converts chunks into embeddings.
4. Stores the embeddings in ChromaDB.
5. Retrieves the most relevant chunks for a user question.
6. Sends only the retrieved context to Gemini.
7. Generates a grounded answer.
8. Provides source chunk IDs.
9. Refuses to answer when the required information is not present in the documents.

The final system is also available through a simple Streamlit web interface.

---

## Project Objective

The objective of Task 5.1 was to build a complete RAG pipeline that can answer questions from private documents while reducing hallucination.

The system was designed to follow this rule:

> Answer using only the retrieved document context. If the information is not present, say that it is not available in the documents.

---

## Dataset

The source document is an AI Job Market technical report containing analysis of:

- 5,773 AI-related job postings
- 5 countries
- 2,152 companies
- 1,159 cities
- Job titles
- Required skills
- Experience levels
- Work arrangements
- Salary information

The report was originally analyzed during Week 1.

---

## RAG Architecture

```text
PDF Document
     |
     v
PDF Text Extraction
     |
     v
Text Chunking
     |
     v
Sentence Transformer
(all-MiniLM-L6-v2)
     |
     v
Embeddings
     |
     v
ChromaDB
     |
     | User Question
     v
Query Embedding
     |
     v
Top-K Similarity Search
     |
     v
Retrieved Chunks
     |
     v
Grounded Prompt
     |
     v
Gemini
     |
     v
Answer + Sources