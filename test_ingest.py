import unittest
import os
import shutil
import sys
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Add the root directory to the Python path to allow importing 'ingest'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import ingest

class TestIngest(unittest.TestCase):

    TEST_PAPERS_PATH = "test_papers/"
    TEST_VECTORSTORE_PATH = "test_vectorstore/"
    TEST_DB_FAISS_PATH = os.path.join(TEST_VECTORSTORE_PATH, "db_faiss")
    TEST_METADATA_PATH = os.path.join(TEST_VECTORSTORE_PATH, "metadata.json")

    def setUp(self):
        """Set up a temporary environment for testing."""
        # Create temporary directories
        os.makedirs(self.TEST_PAPERS_PATH, exist_ok=True)
        os.makedirs(self.TEST_VECTORSTORE_PATH, exist_ok=True)

        # Monkey-patch the paths in the ingest script
        self.original_data_path = ingest.DATA_PATH
        self.original_db_path = ingest.DB_FAISS_PATH
        self.original_meta_path = ingest.METADATA_PATH
        ingest.DATA_PATH = self.TEST_PAPERS_PATH
        ingest.DB_FAISS_PATH = self.TEST_DB_FAISS_PATH
        ingest.METADATA_PATH = self.TEST_METADATA_PATH

        # Store original sys.argv
        self.original_argv = sys.argv

    def tearDown(self):
        """Clean up the temporary environment."""
        # Remove temporary directories
        shutil.rmtree(self.TEST_PAPERS_PATH)
        shutil.rmtree(self.TEST_VECTORSTORE_PATH)

        # Restore original paths
        ingest.DATA_PATH = self.original_data_path
        ingest.DB_FAISS_PATH = self.original_db_path
        ingest.METADATA_PATH = self.original_meta_path

        # Restore original sys.argv
        sys.argv = self.original_argv

    def _create_dummy_pdf(self, filename, content="This is a test page for the ingest script."):
        """Creates a simple PDF file with text content for testing."""
        path = os.path.join(self.TEST_PAPERS_PATH, filename)
        c = canvas.Canvas(path, pagesize=letter)
        c.drawString(72, 720, content)  # Draw string at (1 inch, 10 inches)
        c.save()

    def _get_indexed_sources_from_db(self):
        """Loads the FAISS index and returns the set of indexed source filenames."""
        if not os.path.exists(self.TEST_DB_FAISS_PATH):
            return set()

        embeddings = HuggingFaceEmbeddings(
            model_name=ingest.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
        )
        db = FAISS.load_local(self.TEST_DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return ingest.get_indexed_sources(db)

    def test_full_build_and_append(self):
        """
        Tests the full build and append functionality of the ingest script.
        1. Creates an initial set of documents and builds the index.
        2. Verifies that the correct number of sources are indexed.
        3. Adds a new document and runs the append process.
        4. Verifies that the new document is added and the total count is correct.
        """
        # --- 1. Initial Build ---
        print("\n--- Running Initial Build Test ---")
        self._create_dummy_pdf("doc1.pdf", content="This is the first test document.")
        self._create_dummy_pdf("doc2.pdf", content="This is the second test document.")

        # Simulate running `python ingest.py`
        sys.argv = ['ingest.py']
        ingest.main()

        # Verification for initial build
        indexed_sources = self._get_indexed_sources_from_db()
        self.assertEqual(len(indexed_sources), 2, "Should have indexed 2 documents.")
        self.assertIn("doc1.pdf", indexed_sources)
        self.assertIn("doc2.pdf", indexed_sources)

        # --- 2. Append ---
        print("\n--- Running Append Test ---")
        self._create_dummy_pdf("doc3.pdf", content="This is the third test document.")

        # Simulate running `python ingest.py --append`
        sys.argv = ['ingest.py', '--append']
        ingest.main()

        # Verification for append
        indexed_sources_after_append = self._get_indexed_sources_from_db()
        self.assertEqual(len(indexed_sources_after_append), 3, "Should have indexed 3 documents after append.")
        self.assertIn("doc3.pdf", indexed_sources_after_append)

        # --- 3. No New Docs Append ---
        print("\n--- Running Append Test With No New Docs ---")
        # Simulate running `python ingest.py --append` again
        sys.argv = ['ingest.py', '--append']
        ingest.main()

        # Verification for append with no new docs
        indexed_sources_after_second_append = self._get_indexed_sources_from_db()
        self.assertEqual(len(indexed_sources_after_second_append), 3, "Should still have 3 documents after a redundant append.")


if __name__ == '__main__':
    unittest.main()