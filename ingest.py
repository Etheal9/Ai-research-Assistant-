
import re
import PyPDF2
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "papers/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# 1. Clean the text from PDF artifacts
def clean_text(text):
    text = re.sub(r'^(.*?)\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'^\s*[a-zA-Z]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text

# 2. Deduplicate chunks by content hash
def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        content_hash = hash(chunk.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_chunks.append(chunk)
    return unique_chunks

def extract_pdf_metadata(pdf_path):
    """Extracts title and author from PDF content (first page)."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            
            # First try to get metadata from PDF info
            info = reader.metadata
            title = info.title if info.title else ""
            author = info.author if info.author else ""
            
            # If title is missing, extract from first page content
            if not title and len(reader.pages) > 0:
                first_page = reader.pages[0]
                page_text = first_page.extract_text()
                
                # Extract title using pattern matching
                # Look for lines that appear to be titles (centered, bold, etc.)
                lines = page_text.split('\n')
                
                # Common title patterns in academic papers:
                # 1. Lines that are not too long (3-15 words)
                # 2. Lines that don't start with numbers or special chars
                # 3. Lines that are followed by author names
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    words = line.split()
                    
                    # Skip empty lines, page numbers, headers/footers
                    if (not line or
                        len(words) < 2 or
                        len(words) > 20 or
                        line.isdigit() or
                        any(word.lower() in ['abstract', 'introduction', 'proceedings'] for word in words) or
                        any(char in line for char in ['©', '§', '•']) or
                        re.match(r'^\d+$', line) or  # Page numbers
                        re.match(r'^[A-Z\s]+$', line)):  # ALL CAPS headers
                        continue
                    
                    # This might be a title candidate
                    title = line
                    
                    # Check if next line might be authors
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if (len(next_line.split()) <= 8 and
                            any(word.lower() in ['university', 'department', 'laboratory', 'institute'] for word in next_line.split()) or
                            re.search(r'@|\.edu|\.ac\.', next_line)):
                            author = next_line
                    
                    break  # Found a candidate, break early
            
            return {"title": title, "author": author}
    except Exception as e:
        print(f"Could not extract metadata from {pdf_path}: {e}")
        return {"title": "", "author": ""}

# Load documents
print("Loading documents...")
loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# 3. Extract and print metadata for each document
print("Extracting and displaying metadata for each document...")
for doc in documents:
    print(f"Document metadata: {doc.metadata}")

    # After loading documents, enrich their metadata
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

# 4. Assign metadata to each chunk (already handled by LangChain split_documents)
# But you can enrich or check metadata here if needed
for chunk in text_chunks:
    # Example: Add a custom field if missing
    if 'source' not in chunk.metadata:
        chunk.metadata['source'] = 'unknown'

# 5. Deduplicate chunks
print("Deduplicating chunks...")
text_chunks = deduplicate_chunks(text_chunks)
print(f"{len(text_chunks)} unique chunks after deduplication.")

# Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

# 6. Store original text + metadata with vectors (handled by FAISS.from_documents)
print("Creating and saving the vector store...")
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS_PATH)
print("Vector store rebuilt successfully with cleaned data and metadata.")
