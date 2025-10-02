import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment and basic logging
load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- CONFIG ---
DB_FAISS_PATH = "vectorstore/db_faiss"
PROMPT_PATH = Path("prompts/rag_prompt.txt")

# Initialize LLM and Embeddings (keep consistent with ingest.py)
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

# Safe DB load
try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    logging.info("Loaded FAISS vector store from %s", DB_FAISS_PATH)
except Exception as e:
    logging.warning("Failed to load FAISS index at %s: %s", DB_FAISS_PATH, e)
    db = None

retriever = db.as_retriever(search_kwargs={"k": 5}) if db else None

# Get source docs safely
def get_source_document_list(db_obj):
    if not db_obj:
        return []
    values = []
    try:
        docstore = getattr(db_obj, "docstore", None)
        if docstore and hasattr(docstore, "_dict"):
            values = list(docstore._dict.values())
        else:
            # try iterate stored docs if public API exists
            if hasattr(db_obj, "get_all_documents"):
                values = list(db_obj.get_all_documents())
    except Exception:
        values = []

    unique_sources = {}
    for doc in values:
        try:
            src = doc.metadata.get("source", "Unknown")
            src_file = os.path.basename(src)
            title = doc.metadata.get("title", "") or os.path.splitext(src_file)[0]
            unique_sources[src_file] = title.strip()
        except Exception:
            continue
    return sorted(list(unique_sources.values()))

SOURCE_DOCUMENTS = get_source_document_list(db)

# Load RAG prompt from external file (single responsibility)
if PROMPT_PATH.exists():
    rag_prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
else:
    logging.error("Prompt file missing at %s", PROMPT_PATH)
    rag_prompt_text = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

rag_prompt = PromptTemplate(template=rag_prompt_text, input_variables=["context", "question"])

# Helper: format docs into context
def format_docs(docs):
    if not docs:
        return ""
    parts = []
    for d in docs:
        content = getattr(d, "page_content", "") or ""
        src = d.metadata.get("source", "")
        title = d.metadata.get("title", "")
        header = f"Source: {os.path.basename(src)} | Title: {title}".strip()
        parts.append(header + "\n\n" + content)
    return "\n\n---\n\n".join(parts)

# Greeting handler
def handle_greeting(_question):
    return "Hello — I'm an assistant over a corpus of research on Multi-Agent AI Systems. Ask me about the papers or request a list of available documents."

# Meta handler
def handle_meta_query(_question):
    response = f"I have access to {len(SOURCE_DOCUMENTS)} research papers. Here are the documents in my knowledge base:\n\n"
    for i, title in enumerate(sorted(SOURCE_DOCUMENTS), 1):
        response += f"{i}. {title}\n"
    response += "\nYou can ask me questions about any of these papers."
    return response

# RAG answer flow (explicit retrieval + LLM call)
def answer_rag_question(question):
    if retriever is None:
        return "I don't have document context available right now.", []

    try:
        docs = retriever.get_relevant_documents(question)
    except Exception as e:
        logging.error("Retriever failed: %s", e)
        return "I could not retrieve documents right now.", []

    logging.info("Retrieved %d documents for query.", len(docs))
    if not docs:
        return "I'm sorry — the information does not appear in the available documents.", []

    # debug preview
    try:
        preview = docs[0].page_content[:300].replace("\n", " ")
        logging.debug("Preview: %s", preview)
    except Exception:
        pass

    context = format_docs(docs)
    prompt_text = rag_prompt.format(context=context, question=question)

    # Try multiple LLM invocation patterns defensively
    raw = None
    try:
        raw = llm.invoke(prompt_text)
    except Exception:
        try:
            gen = llm.generate([prompt_text])
            raw = gen
        except Exception as e:
            logging.error("LLM invocation failed: %s", e)
            return "I could not generate an answer right now.", docs

    # Normalize result robustly (handle str, BaseMessage/AIMessage, .text() method, and generation objects)
    answer = None
    if isinstance(raw, str):
        answer = raw.strip()
    else:
        # 1) common chat message: use .content if present
        content = getattr(raw, "content", None)
        if isinstance(content, str) and content.strip():
            answer = content.strip()
        else:
            # 2) .text may be a method or attribute
            text_attr = getattr(raw, "text", None)
            if callable(text_attr):
                try:
                    answer = str(text_attr()).strip()
                except Exception:
                    answer = None
            elif isinstance(text_attr, str) and text_attr.strip():
                answer = text_attr.strip()
            else:
                # 3) generation-like object
                gens = getattr(raw, "generations", None)
                if gens:
                    try:
                        answer = gens[0][0].text.strip()
                    except Exception:
                        answer = None
                # 4) fallback to str()
                if not answer:
                    answer = str(raw).strip()

    return answer, docs

# Chain-like interface for compatibility with other code
class RagChain:
    def invoke(self, question):
        ans, docs = answer_rag_question(question)
        return {"answer": ans, "documents": docs}

rag_chain = RagChain()

# Router with greeting detection and meta detection
def query_router(question):
    q = (question or "").lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(q == g or q.startswith(g + " ") for g in greetings):
        return "greet"
    meta_keywords = ["what documents", "what papers", "list your sources", "what are your sources", "what resources", "your knowledge base", "list papers"]
    if any(k in q for k in meta_keywords):
        return "meta_query"
    return "rag_query"

# --- MAIN INTERACTION LOOP ---
if __name__ == "__main__":
    print("Welcome to the AI Research Synthesizer!")
    print(f"Knowledge base loaded with {len(SOURCE_DOCUMENTS)} documents.")
    print("Type 'exit' to quit.")

    try:
        while True:
            user_question = input("\nYour question: ").strip()
            if not user_question:
                continue
            if user_question.lower() == "exit":
                break

            route = query_router(user_question)

            if route == "greet":
                print("Answer:", handle_greeting(user_question))
                continue

            if route == "meta_query":
                print("Answer:", handle_meta_query(user_question))
                continue

            answer, docs = answer_rag_question(user_question)
            print("Answer:", answer)
            if docs:
                unique_sources = {}
                for doc in docs:
                    try:
                        source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                        display_name = doc.metadata.get("title", source_file) or source_file
                        unique_sources[source_file] = display_name
                    except Exception:
                        continue
                if unique_sources:
                    print("\nSources:")
                    for display_name in sorted(unique_sources.values()):
                        print(f"- {display_name}")
    except KeyboardInterrupt:
        print("\nExiting.")