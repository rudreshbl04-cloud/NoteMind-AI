"""Unit tests for NoteMind AI RAG logic, grounding, thresholding, and citations."""

import unittest
from unittest.mock import MagicMock, patch
from src.rag import (
    format_context_and_sources,
    answer_question,
    NOT_FOUND_MESSAGE,
    MAX_RELEVANCE_DISTANCE,
)


class TestRagPipeline(unittest.TestCase):
    """Test suite for RAG retrieval formatting, grounding, and citations."""

    def test_format_context_and_sources(self):
        chunks = [
            {"text": "Python was created in 1991.", "source": "ch1.pdf", "page": 2, "distance": 0.1},
            {"text": "Python emphasizes code readability.", "source": "ch1.pdf", "page": 2, "distance": 0.2},
            {"text": "Guido van Rossum released Python.", "source": "ch2.pdf", "page": 5, "distance": 0.25},
        ]
        context, sources = format_context_and_sources(chunks)
        self.assertIn("Python was created in 1991", context)
        self.assertIn("Guido van Rossum", context)
        # Should have deduplicated to 2 unique source citations
        self.assertEqual(len(sources), 2)
        displays = [s["display"] for s in sources]
        self.assertIn("ch1.pdf — Page 2", displays)
        self.assertIn("ch2.pdf — Page 5", displays)

    def test_empty_question_returns_prompt_guide(self):
        mock_client = MagicMock()
        res = answer_question("", mock_client)
        self.assertFalse(res["found"])
        self.assertIn("Please ask a question", res["answer"])
        self.assertEqual(res["sources"], [])

    @patch("src.rag.get_total_chunk_count", return_value=0)
    def test_no_notes_indexed_returns_guidance(self, mock_count):
        mock_client = MagicMock()
        res = answer_question("Who made Python?", mock_client)
        self.assertFalse(res["found"])
        self.assertIn("No notes have been uploaded yet", res["answer"])

    @patch("src.rag.get_total_chunk_count", return_value=5)
    @patch("src.rag.get_text_embedding", return_value=[0.1, 0.2, 0.3])
    @patch("src.rag.search_similar_chunks")
    def test_distance_threshold_filters_out_irrelevant_chunks(self, mock_search, mock_emb, mock_count):
        # Chunks with distance > MAX_RELEVANCE_DISTANCE (e.g. 0.85 > 0.65)
        mock_search.return_value = [
            {"text": "Cooking pasta recipe.", "source": "recipes.pdf", "page": 4, "distance": 0.88}
        ]
        mock_client = MagicMock()
        res = answer_question("What is the capital of France?", mock_client, distance_threshold=0.65)
        self.assertEqual(res["answer"], NOT_FOUND_MESSAGE)
        self.assertFalse(res["found"])
        self.assertEqual(res["sources"], [])

    @patch("src.rag.get_total_chunk_count", return_value=5)
    @patch("src.rag.get_text_embedding", return_value=[0.1, 0.2, 0.3])
    @patch("src.rag.search_similar_chunks")
    @patch("src.rag.generate_answer", return_value="Python was created by Guido van Rossum.")
    def test_grounded_answer_with_accurate_citations(self, mock_gen, mock_search, mock_emb, mock_count):
        mock_search.return_value = [
            {
                "text": "Python was created by Guido van Rossum.",
                "source": "history.pdf",
                "page": 3,
                "distance": 0.15,
            }
        ]
        mock_client = MagicMock()
        res = answer_question("Who created Python?", mock_client)

        self.assertTrue(res["found"])
        self.assertEqual(res["answer"], "Python was created by Guido van Rossum.")
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["source"], "history.pdf")
        self.assertEqual(res["sources"][0]["page"], 3)
        self.assertEqual(res["sources"][0]["display"], "history.pdf — Page 3")

    @patch("src.rag.get_total_chunk_count", return_value=5)
    @patch("src.rag.get_text_embedding", return_value=[0.1, 0.2, 0.3])
    @patch("src.rag.search_similar_chunks")
    @patch("src.rag.generate_answer", return_value="Not found in your notes.")
    def test_model_not_found_handling(self, mock_gen, mock_search, mock_emb, mock_count):
        mock_search.return_value = [
            {
                "text": "Some vaguely related topic.",
                "source": "general.pdf",
                "page": 1,
                "distance": 0.40,
            }
        ]
        mock_client = MagicMock()
        res = answer_question("Who walked on the moon?", mock_client)

        self.assertFalse(res["found"])
        self.assertEqual(res["answer"], NOT_FOUND_MESSAGE)
        self.assertEqual(res["sources"], [])


if __name__ == "__main__":
    unittest.main()
