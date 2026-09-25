"""Unit and integration tests for ChromaDB vector store in NoteMind AI."""

import os
import shutil
import unittest
import chromadb
from src.vector_store import (
    add_chunks_to_vector_store,
    search_similar_chunks,
    reset_vector_store,
    get_indexed_files,
    get_total_chunk_count,
    COLLECTION_NAME,
)

TEST_DB_PATH = os.path.join("data", "test_chroma")


class TestVectorStore(unittest.TestCase):
    """Test suite for local ChromaDB operations."""

    def setUp(self):
        os.makedirs(TEST_DB_PATH, exist_ok=True)
        self.client = chromadb.PersistentClient(path=TEST_DB_PATH)
        # Clear collection before each test
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    def tearDown(self):
        try:
            shutil.rmtree(TEST_DB_PATH, ignore_errors=True)
        except Exception:
            pass

    def test_add_and_search_chunks(self):
        chunks = [
            {"text": "Python was created by Guido van Rossum.", "source": "python_history.pdf", "page": 1},
            {"text": "JavaScript was developed by Brendan Eich at Netscape.", "source": "js_history.pdf", "page": 3},
        ]
        # Two dummy 4-dimensional normalized vectors
        embeddings = [
            [0.1, 0.9, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ]

        added_count = add_chunks_to_vector_store(chunks, embeddings, client=self.client)
        self.assertEqual(added_count, 2)
        self.assertEqual(get_total_chunk_count(self.client), 2)

        files = get_indexed_files(self.client)
        self.assertIn("python_history.pdf", files)
        self.assertIn("js_history.pdf", files)

        # Search query closely aligned with Python chunk
        query_emb = [0.1, 0.85, 0.0, 0.0]
        results = search_similar_chunks(query_emb, top_k=1, client=self.client)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "python_history.pdf")
        self.assertEqual(results[0]["page"], 1)
        self.assertIn("Guido van Rossum", results[0]["text"])

    def test_duplicate_prevention(self):
        # Adding same file twice should replace, not duplicate
        chunks = [
            {"text": "Python notes v1", "source": "notes.pdf", "page": 1},
        ]
        embeddings = [[0.5, 0.5, 0.5, 0.5]]
        add_chunks_to_vector_store(chunks, embeddings, client=self.client)
        self.assertEqual(get_total_chunk_count(self.client), 1)

        # Re-index same file with updated text
        chunks_v2 = [
            {"text": "Python notes v2 updated", "source": "notes.pdf", "page": 1},
        ]
        embeddings_v2 = [[0.5, 0.5, 0.5, 0.5]]
        add_chunks_to_vector_store(chunks_v2, embeddings_v2, client=self.client)
        self.assertEqual(get_total_chunk_count(self.client), 1)

    def test_reset_vector_store(self):
        chunks = [
            {"text": "Sample text", "source": "sample.pdf", "page": 1},
        ]
        embeddings = [[0.1, 0.2, 0.3, 0.4]]
        add_chunks_to_vector_store(chunks, embeddings, client=self.client)
        self.assertEqual(get_total_chunk_count(self.client), 1)

        reset_vector_store(self.client)
        self.assertEqual(get_total_chunk_count(self.client), 0)
        self.assertEqual(get_indexed_files(self.client), [])


if __name__ == "__main__":
    unittest.main()
