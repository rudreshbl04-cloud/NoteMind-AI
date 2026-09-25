"""Unit tests for Gemini client validation and security in NoteMind AI."""

import unittest
from unittest.mock import MagicMock, patch
from google.genai.errors import ClientError, APIError
from src.gemini_client import (
    validate_api_key,
    create_gemini_client,
    get_text_embedding,
    generate_answer,
)


class TestGeminiClient(unittest.TestCase):
    """Test suite for client setup, key validation, and security."""

    def test_empty_or_whitespace_key_validation(self):
        valid, msg = validate_api_key("")
        self.assertFalse(valid)
        self.assertIn("required", msg.lower())

        valid, msg = validate_api_key("    ")
        self.assertFalse(valid)
        self.assertIn("required", msg.lower())

        self.assertIsNone(create_gemini_client(""))
        self.assertIsNone(create_gemini_client("   "))

    @patch("src.gemini_client.create_gemini_client")
    def test_invalid_key_error_handling(self, mock_create):
        mock_client = MagicMock()
        # Mock ClientError with 400 API_KEY_INVALID on models.list
        mock_client.models.list.side_effect = ClientError(
            400,
            {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}},
        )
        mock_create.return_value = mock_client

        valid, msg = validate_api_key("fake_invalid_key")
        self.assertFalse(valid)
        self.assertIn("Invalid Gemini API key", msg)
        # Ensure the test key is NOT exposed in the error message
        self.assertNotIn("fake_invalid_key", msg)

    @patch("src.gemini_client.create_gemini_client")
    def test_network_error_handling(self, mock_create):
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("Failed to connect to host")
        mock_create.return_value = mock_client

        valid, msg = validate_api_key("some_key")
        self.assertFalse(valid)
        self.assertIn("Network error", msg)
        self.assertNotIn("some_key", msg)

    @patch("src.gemini_client.create_gemini_client")
    def test_successful_validation(self, mock_create):
        mock_client = MagicMock()
        dummy_model = MagicMock(name="models/gemini-2.5-flash")
        dummy_model.name = "models/gemini-2.5-flash"
        mock_client.models.list.return_value = [dummy_model]
        mock_create.return_value = mock_client

        valid, msg = validate_api_key("valid_test_key_sample")
        self.assertTrue(valid)
        self.assertIn("Connected successfully", msg)
        self.assertNotIn("valid_test_key_sample", msg)


if __name__ == "__main__":
    unittest.main()
