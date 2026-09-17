from pypdf import PdfReader

pdf_path = "../documents/sample.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    print(f"\n--- PAGE {page_number} ---")
    print(text[:1000])