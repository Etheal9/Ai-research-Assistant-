# AI Research Synthesizer

Short description
- A local Retrieval-Augmented Generation (RAG) assistant for research on Multi‑Agent AI Systems.
- Ingests PDFs, builds a FAISS vectorstore of text chunks (embeddings), and answers user questions using a Groq LLM.

Quick status
- Ingest script: `ingest.py` — build or append to the vectorstore.
- Assistant REPL: `main.py` — query the KB, list available papers, and answer questions.
- RAG prompt: `prompts/rag_prompt.txt` — edit to tune assistant behavior.
- PDF renamer (optional): `rename_pdfs.py` — extracts titles and renames numeric filenames.

Requirements (summary)
- Python 3.10+ recommended.
- See `requirements.txt` for pip packages and example versions.

Quickstart (Windows PowerShell)
1. Create and activate venv:
   powershell> python -m venv .venv
   powershell> .\.venv\Scripts\Activate.ps1

2. Install dependencies:
   powershell> pip install -r requirements.txt

3. (Optional) Dry-run rename to fix numeric filenames:
   powershell> python rename_pdfs.py --dry-run
   If output looks good, run without `--dry-run` to rename files.

4. Build or append vectorstore:
   - Full rebuild (overwrite):
     powershell> python ingest.py
   - Append new papers (must use same embedding model):
     powershell> python ingest.py --append
   - Force rebuild despite embedding mismatch:
     powershell> python ingest.py --force

5. Run the assistant REPL:
   powershell> python main.py

Environment
- Place PDF files in `papers/`.
- Prompt template is `prompts/rag_prompt.txt`.
- Vectorstore is saved under `vectorstore/` (FAISS files) and metadata at `vectorstore/metadata.json`.
- Provide any LLM provider credentials in a `.env` file (example):
  ```
  # example .env entries - replace with your provider's variables
  GROQ_API_KEY=…
  OPENAI_API_KEY=…
  ```

What to check after ingest
- ingest.py prints:
  - number of source PDFs loaded
  - number of chunks split and non-empty chunk count
  - a sample chunk preview
  - final chunk count and metadata saved

If retrieval returns no context
- Re-run `ingest.py` and confirm non-empty chunk counts and sample preview.
- Ensure `EMBEDDING_MODEL_NAME` in `ingest.py` matches the embeddings used in `main.py`.
- If embedding model changed, rebuild (no safe mix).

File layout (important files)
- ingest.py — ingest pipeline, chunking, dedup, save FAISS
- main.py — assistant REPL, retrieval + LLM call flow
- prompts/rag_prompt.txt — RAG prompt template (editable)
- rename_pdfs.py — optional renamer to use PDF title as filename
- vectorstore/ — saved FAISS index and metadata.json

Troubleshooting tips
- "Bound method" output: means LLM returned a message object — main.py now normalizes common response shapes.
- Titles show as numeric filenames: run `rename_pdfs.py` then re-ingest to preserve titles in index metadata.
- If FAISS load fails: inspect `vectorstore/metadata.json` and run `ingest.py` to rebuild.

Production notes (brief)
- For production consider:
  - Managed vector DB (Qdrant, Milvus, Weaviate) for concurrency and persistence.
  - GPU or faster embeddings + batching for latency.
  - Index versioning and atomic rebuilds.
  - Monitoring, timeouts, rate limits, and secure key handling.

License
- Add your project license file if needed (e.g., `LICENSE`).

If you want, I can:
- Add a sample `.env.example`.
- Add a small sanity-check script that queries the index after ingest and prints the retrieved preview.