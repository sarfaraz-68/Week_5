from prompt_builder import build_prompt

chunks = [
    "Python (308) and Computer Vision (276) lead, followed by NLP (171), Azure (133), R (108), SQL (87).",
    "Data Scientist (281) and AI Engineer (270) lead the job-title rankings.",
    "The United States has 1,837 postings."
]

question = "What are the top requested skills in AI jobs?"

prompt = build_prompt(
    chunks,
    question
)

print(prompt)