
import arxiv
import os

# --- ⚙️ CONFIGURATION: EDIT THESE VALUES ---
# 1. Set the topic you want to research
#search_query = 'cat:cs.AI AND "multi-agent system"'
search_query = 'cat:cs.AI AND ("multi-agent system" OR "agent-based modeling" OR "distributed agents")'


# 2. Set the number of papers you want to download
max_results = 50

# 3. Set the folder where PDFs will be saved
DOWNLOAD_PATH = "papers/"
# --- END OF CONFIGURATION ---

# Create the directory if it doesn't exist
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

print(f"Searching for '{search_query}' on arXiv...")

# Perform the search using the arxiv library
# Updated code to use the modern arxiv.Client() to avoid deprecation warnings.
client = arxiv.Client()
search = arxiv.Search(
  query = search_query,
  max_results = max_results,
  sort_by = arxiv.SortCriterion.Relevance
)

# Convert search results to a list to get the total count
results_list = list(search.results())
total_papers = len(results_list)

# --- NEW DEBUGGING STEP ---
# This will tell us exactly how many papers were found before trying to download.
print(f"Found {total_papers} papers matching the query.")

if total_papers == 0:
    # If no papers are found, print a helpful message.
    print("\n--- DIAGNOSIS ---")
    print("The search returned 0 results. This is why no downloads started.")
    print("\nPossible solutions:")
    print("1. Your 'search_query' might be too specific. Try broadening it.")
    print("   Example: `search_query = 'multi-agent systems'`")
    print("2. There could be a temporary network issue connecting to arXiv.")
    print("\nPlease check the `search_query` variable in this script and try again.")
else:
    # If papers are found, proceed with the download process.
    print("Starting download process...")
    successful_downloads = 0
    failed_downloads = []

    for i, result in enumerate(results_list):
        pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
        file_path = os.path.join(DOWNLOAD_PATH, pdf_filename)

        print(f"\nProcessing paper {i+1}/{total_papers}: '{result.title}'")

        if os.path.exists(file_path):
            print("--- INFO: File already exists. Skipping. ---")
            continue

        try:
            result.download_pdf(dirpath=DOWNLOAD_PATH, filename=pdf_filename)
            print(f"--- SUCCESS: Downloaded '{pdf_filename}' ---")
            successful_downloads += 1
        except Exception as e:
            print(f"--- ERROR: Could not download '{result.title}'. Skipping. ---")
            print(f"   Reason: {e}")
            failed_downloads.append(result.title)

    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Successfully downloaded: {successful_downloads} papers.")
    print(f"Failed to download: {len(failed_downloads)} papers.")
    if failed_downloads:
        print("\nFailed titles:")
        for title in failed_downloads:
            print(f"- {title}")
    print("="*50)