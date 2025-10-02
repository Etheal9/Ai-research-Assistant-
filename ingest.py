import re
import PyPDF2
import os
import json
import time
import hashlib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "papers/"
DB_FAISS_PATH = "vectorstore/db_faiss"
METADATA_PATH = "vectorstore/metadata.json"

# Embedding model name used for this ingest run (bump when you change model)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Clean the text from PDF artifacts
def clean_text(text):
    if not text:
        return text
    text = re.sub(r'\[\s*\d+(?:[,\s\d]*)\s*\]', '', text)
    text = re.sub(r'(?m)^[A-Za-z]\s*$', '', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

# 2. Deduplicate chunks by content hash (stable)
def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        content = getattr(chunk, "page_content", "") or ""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(chunk)
    return unique_chunks

def extract_pdf_metadata(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            info = reader.metadata
            title = info.title if info and getattr(info, "title", None) else ""
            author = info.author if info and getattr(info, "author", None) else ""
            if not title and len(reader.pages) > 0:
                first_page = reader.pages[0]
                page_text = first_page.extract_text() or ""
                lines = page_text.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    words = line.split()
                    if len(words) < 2 or len(words) > 20:
                        continue
                    if re.search(r'^(abstract|introduction|references)$', line.strip().lower()):
                        continue
                    title = line
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if re.search(r'@|\.edu|\.ac\.', next_line) or any(w.lower() in next_line.lower() for w in ['university','institute','department']):
                            author = next_line
                    break
            return {"title": title, "author": author}
    except Exception as e:
        print(f"Could not extract metadata from {pdf_path}: {e}")
        return {"title": "", "author": ""}

# metadata helpers
def save_index_metadata(path, model_name, chunk_count):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"embedding_model": model_name, "created_at": int(time.time()), "chunk_count": chunk_count}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_index_metadata(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Load documents
print("Loading documents...")
loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Extract and display metadata
print("Extracting detailed PDF metadata (title, author)...")
for doc in documents:
    source_path = doc.metadata.get("source", None)
    if source_path:
        meta = extract_pdf_metadata(source_path)
        doc.metadata.update(meta)
        print(f"{source_path} | Title: {meta['title']} | Author: {meta['author']}")

# Clean the documents before splitting
print("Cleaning document text...")
for doc in documents:
    doc.page_content = clean_text(doc.page_content)

# Split documents into chunks
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
text_chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(text_chunks)} chunks.")

# Quick debug: how many chunks actually contain useful text?
non_empty = [c for c in text_chunks if c.page_content and c.page_content.strip()]
print(f"{len(non_empty)} non-empty chunks (out of {len(text_chunks)})")
if non_empty:
    print("Sample chunk preview:")
    print(non_empty[0].page_content[:400].replace('\\n', ' ') + "...")

# Ensure chunk metadata has source
for chunk in text_chunks:
    if 'source' not in chunk.metadata:
        chunk.metadata['source'] = 'unknown'

# Deduplicate
print("Deduplicating chunks...")
text_chunks = deduplicate_chunks(non_empty)
print(f"{len(text_chunks)} unique chunks after deduplication.")

# Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

# warn on existing meta mismatch
existing_meta = load_index_metadata(METADATA_PATH)
if existing_meta and existing_meta.get("embedding_model") != EMBEDDING_MODEL_NAME:
    print("WARNING: existing index metadata indicates a different embedding model was used:")
    print(f"  existing: {existing_meta.get('embedding_model')}")
    print(f"  current:  {EMBEDDING_MODEL_NAME}")
    print("Vectors are not compatible across embedding models. Rebuilding the index is recommended.")

# Create and save
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
print("Creating and saving the vector store...")
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
save_index_metadata(METADATA_PATH, EMBEDDING_MODEL_NAME, len(text_chunks))
print("Vector store rebuilt successfully with cleaned data and metadata.")
