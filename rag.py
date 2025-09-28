import os
import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM  # ✅ new imports

pdf_path = input("Enter the full path of your PDF (or just filename if in same folder): ").strip()

# Normalize (remove quotes, expand ~)
pdf_path = pdf_path.strip('"').strip("'")
pdf_path = os.path.expanduser(pdf_path)

# If no extension, add .pdf
if not pdf_path.lower().endswith(".pdf"):
    pdf_path += ".pdf"

# Use pathlib to handle special characters
pdf_path = str(pathlib.Path(pdf_path))

# Check existence
if not os.path.exists(pdf_path):
    print(f"❌ File not found: {pdf_path}")
    exit(1)

# Load PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Build vector database
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma.from_documents(chunks, embeddings)

# Load LLM
llm = OllamaLLM(model="gemma3:1b")

# Main Q&A loop
while True:
    query = input("\nEnter your question about the PDF (or type 'exit' to quit): ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Exiting. Goodbye!")
        break

    docs = db.similarity_search(query, k=3)

    if not docs:
        print("⚠️ No relevant content found in the PDF for your query.")
        continue

    context = " ".join([d.page_content for d in docs])
    prompt = f"Answer based on this context:\n\n{context}\n\nQuestion: {query}"
    response = llm.invoke(prompt)  # ✅ updated to use invoke()

    print("\n📌 Answer:")
    print(response)
