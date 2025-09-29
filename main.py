import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# --- SETUP AND LOAD KNOWLEDGE BASE ---
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize LLM and Embeddings
llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

# Load the vector store from disk
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={'k': 5}) # Increased k for a larger KB

# --- **IMPROVED**: DYNAMICALLY GET ALL SOURCE DOCUMENT TITLES ---
# This function reads all the metadata from the vector store to create a comprehensive,
# up-to-date list of all the source documents with their proper titles.
def get_source_document_list(db):
    print("Extracting document information from knowledge base...")
    
    # Get all documents from the vector store
    all_docs = db.docstore._dict.values()
    unique_sources = {} # Use a dictionary to handle duplicates by source file
    
    for doc in all_docs:
        # Get the source filename
        source_path = doc.metadata.get("source", "Unknown")
        source_file = os.path.basename(source_path)
        
        # Try to get the title from metadata, prefer 'title' field
        title = doc.metadata.get("title", "")
        
        # If no title in metadata, try to extract from content or use filename
        if not title or title.strip() == "":
            # Use filename without extension as fallback
            title = os.path.splitext(source_file)[0]
        
        # Store with source file as key to avoid duplicates
        if source_file not in unique_sources or len(title) > len(unique_sources[source_file]):
            unique_sources[source_file] = title.strip()
    
    print(f"Found {len(unique_sources)} unique documents in the knowledge base.")
    
    # Return sorted list of titles
    return sorted(list(unique_sources.values()))

# Create the dynamic list of source documents
SOURCE_DOCUMENTS = get_source_document_list(db)

# --- RAG CHAIN (For Domain-Specific Questions) ---
def format_docs(docs):
    """Converts a list of Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt_template = """

You are a helpful AI assistant with expertise in Multi-Agent AI Systems. You have access to research documents on this topic.

If the user's question is related to research documents you have, Multi-Agent AI Systems, Reinforcement Learning, or Generative Agents, use the provided context to give a comprehensive answer. If the context contains relevant information, prioritize it in your response.

If the question is a general greeting, casual conversation, or unrelated to your expertise area, respond naturally and helpfully as a friendly AI assistant that you can not answer out of the unrelated to your expertise area.

use a Funnel‑style Chain of Thought when answering.
This means:

1. Start broad by introducing the general topic.
2. Narrow down into sub‑topics step by step, each building on the previous.
3. Highlight key details and examples at each stage.
4. Conclude with a clear synthesis and 2–3 practical takeaways.

Never jump straight into specifics without context.
If the user asks a complex question, break it into smaller reasoning steps before answering.
Keep your tone professional but approachable, and explain technical terms with analogies when possible.

Context: {context}
Question: {question}

Helpful Answer:
"""


# **NEW**: Simplified and corrected RAG chain
# First convert string input to dictionary, then process
def string_to_dict(question):
    """Convert string question to dictionary format."""
    return {"question": question}

# Define the PromptTemplate for the RAG prompt
rag_prompt = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "question"]
)

rag_chain = (
    string_to_dict
    | RunnablePassthrough.assign(documents=lambda x: retriever.invoke(x["question"]))
    | RunnablePassthrough.assign(
        answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["documents"]))
            | rag_prompt
            | llm
            | StrOutputParser()
        )
    )
)

# --- ROUTER AND META-QUERY HANDLER ---
def handle_meta_query(question):
    """Answers questions about the knowledge base using the dynamic list of sources."""
    response = f"I have access to {len(SOURCE_DOCUMENTS)} research papers on Multi-Agent AI Systems, Reinforcement Learning, and Generative Agents. Here are all the documents in my knowledge base:\n\n"
    
    # Add each document with proper formatting
    for i, doc_title in enumerate(sorted(SOURCE_DOCUMENTS), 1):
        response += f"{i}. {doc_title}\n"
    
    response += f"\nYou can ask me questions about any of these papers or topics related to Multi-Agent AI Systems!"
    return response

def query_router(question):
    """Classifies a user's question to decide which tool to use."""
    question_lower = question.lower()
    
    # Expanded keywords for meta-queries about the knowledge base and resources
    meta_keywords = [
        "what documents", "what papers", "list your sources", "what are your sources",
        "what resources", "list resources", "what pdfs", "list pdfs", 
        "show me your papers", "show me documents", "what research papers",
        "list papers", "list documents", "show sources", "available papers",
        "available documents", "what data", "your knowledge base", "knowledge base",
        "bibliography", "references", "what studies", "research sources",
        "show me your sources", "list all sources", "all sources", "your sources",
        "show all documents", "show all papers", "list all papers", "list all documents",
        "what do you have", "sources you have", "papers you have", "documents you have"
    ]
    
    # Check for direct meta-questions
    if any(keyword in question_lower for keyword in meta_keywords):
        return "meta_query"
    
    # For everything else, use the RAG chain (it will handle both domain and general questions)
    return "rag_query"

# --- MAIN INTERACTION LOOP ---
if __name__ == "__main__":
    print("Welcome to the AI Research Synthesizer!")
    print(f"Knowledge base loaded with {len(SOURCE_DOCUMENTS)} documents.")
    print("Ask a question about Multi-Agent AI Systems or about the knowledge base. Type 'exit' to quit.")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break
        
        route = query_router(user_question)
        
        if route == "meta_query":
            answer = handle_meta_query(user_question)
            print("Answer:", answer)
        
        elif route == "rag_query":
            result = rag_chain.invoke(user_question)
            print("Answer:", result["answer"])
            
            source_documents = result["documents"]
            unique_sources = {}
            for doc in source_documents:
                source_file = os.path.basename(doc.metadata.get("source", "Unknown Source"))
                display_name = doc.metadata.get("title", source_file)
                if not display_name:
                    display_name = source_file
                unique_sources[source_file] = display_name

            if unique_sources:
                print("\nSources:")
                for display_name in sorted(unique_sources.values()):
                    print(f"- {display_name}")