import re
import PyPDF2
import os
import json
import time
import hashlib
import argparse
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

def get_indexed_sources(db):
    """Get a set of already indexed source file basenames."""
    indexed_sources = set()
    if hasattr(db, "docstore") and hasattr(db.docstore, "_dict"):
        for doc in db.docstore._dict.values():
            source = doc.metadata.get("source")
            if source:
                indexed_sources.add(os.path.basename(source))
    return indexed_sources

def process_documents(documents):
    """Shared processing pipeline for new documents."""
    print("Extracting detailed PDF metadata (title, author)...")
    for doc in documents:
        source_path = doc.metadata.get("source")
        if source_path:
            meta = extract_pdf_metadata(source_path)
            doc.metadata.update(meta)
            print(f"  - {source_path} | Title: {meta.get('title', 'N/A')}")

    print("Cleaning document text...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(text_chunks)} chunks.")

    non_empty = [c for c in text_chunks if c.page_content and c.page_content.strip()]
    if not non_empty:
        return []

    print(f"{len(non_empty)} non-empty chunks (out of {len(text_chunks)})")
    print("Deduplicating chunks...")
    unique_chunks = deduplicate_chunks(non_empty)
    print(f"{len(unique_chunks)} unique chunks after deduplication.")
    return unique_chunks

def build_index():
    """Build the vector store from scratch."""
    print("\n--- Starting full vector store rebuild ---\n")

    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    documents = loader.load()

    if not documents:
        print("No documents found in papers/. Exiting.")
        return

    print(f"Loaded {len(documents)} documents.")
    text_chunks = process_documents(documents)

    if not text_chunks:
        print("No valid content could be processed from documents. Aborting.")
        return

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})

    print("Creating and saving the vector store...")
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(text_chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    save_index_metadata(METADATA_PATH, EMBEDDING_MODEL_NAME, len(text_chunks))
    print("\n--- Vector store rebuilt successfully ---\n")

def append_to_index():
    """Append new documents to an existing vector store."""
    print("\n--- Starting vector store append ---\n")

    if not os.path.exists(DB_FAISS_PATH) or not os.path.exists(METADATA_PATH):
        print("No existing vector store found. Please run a full build first.")
        return

    existing_meta = load_index_metadata(METADATA_PATH)
    if not existing_meta:
        print("Could not load metadata. Rebuilding is recommended.")
        return

    if existing_meta.get("embedding_model") != EMBEDDING_MODEL_NAME:
        print("ERROR: New embedding model is different from the one in the existing index.")
        print(f"  - Existing: {existing_meta.get('embedding_model')}")
        print(f"  - Current:  {EMBEDDING_MODEL_NAME}")
        print("Appending with different models is not supported. Please rebuild the index (run without --append).")
        return

    print("Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    indexed_sources = get_indexed_sources(db)
    print(f"Found {len(indexed_sources)} already indexed sources.")

    all_pdf_files = {f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')}
    new_pdf_files = sorted(list(all_pdf_files - indexed_sources))

    if not new_pdf_files:
        print("No new documents to add.")
        return

    print(f"Found {len(new_pdf_files)} new documents to ingest:")
    for f in new_pdf_files:
        print(f"  - {f}")

    new_docs = []
    for doc_file in new_pdf_files:
        loader = PyPDFLoader(os.path.join(DATA_PATH, doc_file))
        new_docs.extend(loader.load())

    text_chunks = process_documents(new_docs)

    if not text_chunks:
        print("No new valid chunks to add after processing. Exiting.")
        return

    print("Adding new documents to the vector store...")
    db.add_documents(text_chunks)

    db.save_local(DB_FAISS_PATH)
    total_chunks = len(db.docstore._dict)
    save_index_metadata(METADATA_PATH, EMBEDDING_MODEL_NAME, total_chunks)
    print(f"\n--- Successfully appended {len(text_chunks)} new chunks. ---")
    print(f"--- Vector store now contains {total_chunks} total chunks. ---\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into a FAISS vector store.")
    parser.add_argument("--append", action="store_true", help="Append to existing vector store instead of rebuilding.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if embedding models mismatch.")
    args = parser.parse_args()

    if args.append:
        if args.force:
            print("Warning: --force is not used with --append. Model compatibility is always checked.")
        append_to_index()
    else:
        existing_meta = load_index_metadata(METADATA_PATH)
        if existing_meta and existing_meta.get("embedding_model") != EMBEDDING_MODEL_NAME:
            print("WARNING: Embedding model has changed since the last build.")
            print(f"  - Existing: {existing_meta.get('embedding_model')}")
            print(f"  - Current:  {EMBEDDING_MODEL_NAME}")
            if args.force:
                print("Proceeding with rebuild due to --force.\n")
                build_index()
            else:
                print("Aborting. Use --force to overwrite the existing index with the new model.\n")
        else:
            build_index()

if __name__ == "__main__":
    main()
