"""Integration and UI tests for NoteMind AI using Streamlit AppTest."""

import os
import unittest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


class TestNoteMindApp(unittest.TestCase):
    """Test suite for app.py UI states, screen transitions, and interactions."""

    def test_setup_screen_initial_state(self):
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()


        # Check title & description presence
        self.assertFalse(at.exception)
        self.assertEqual(len(at.text_input), 1)
        api_input = at.text_input[0]
        self.assertEqual(api_input.label, "Gemini API Key")
        # In Streamlit protobuf, proto.type == 1 corresponds to PASSWORD input mode
        self.assertEqual(api_input.proto.type, 1)

        # Check Connect button
        connect_buttons = [b for b in at.button if b.label == "Connect"]
        self.assertEqual(len(connect_buttons), 1)

    def test_empty_key_submission(self):
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()

        # Click connect with empty input
        at.button[0].click().run()
        self.assertFalse(at.exception)

        # Should show warning
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("required" in w.lower() for w in warnings))

    @patch("src.gemini_client.validate_api_key")
    def test_invalid_key_submission(self, mock_validate):
        mock_validate.return_value = (False, "Invalid Gemini API key. Please check the key and try again.")
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()

        at.text_input[0].input("bad_key_12345")
        at.button[0].click().run()

        errors = [e.value for e in at.error]
        self.assertTrue(any("Invalid Gemini API key" in e for e in errors))

    @patch("src.gemini_client.validate_api_key")
    def test_successful_connect_and_screen_routing(self, mock_validate):
        mock_validate.return_value = (True, "Connected successfully!")
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        at.run()

        at.text_input[0].input("valid_dummy_key")
        at.button[0].click().run()

        # Check that session state updated
        self.assertEqual(at.session_state["gemini_api_key"], "valid_dummy_key")

        # Now running the app in authenticated state should show Screen 2
        disconnect_buttons = [b for b in at.sidebar.button if "Disconnect" in b.label]
        self.assertTrue(len(disconnect_buttons) >= 1)

    def test_disconnect_button_returns_to_setup_screen(self):
        at = AppTest.from_file(APP_PATH, default_timeout=10)
        # Pre-seed session state with a key
        at.session_state["gemini_api_key"] = "test_key"
        at.session_state["gemini_client"] = MagicMock()
        at.run()

        # Verify we are on Screen 2
        self.assertTrue(any("Disconnect" in b.label for b in at.sidebar.button))

        # Click Disconnect
        disconnect_btn = [b for b in at.sidebar.button if "Disconnect" in b.label][0]
        disconnect_btn.click().run()

        # Verify key was wiped from session state and back to Screen 1
        self.assertIsNone(at.session_state.get("gemini_api_key"))
        self.assertTrue(any(b.label == "Connect" for b in at.button))


if __name__ == "__main__":
    unittest.main()
