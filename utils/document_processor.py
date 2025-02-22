from PyPDF2 import PdfReader
import textract

def parse_uploaded_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files"""
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in reader.pages])
    elif uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = textract.process(uploaded_file).decode("utf-8")
    return text