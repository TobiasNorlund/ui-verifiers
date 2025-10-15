"""Tests for ActionDecoder class."""

import pytest
from src.parsers.action_decoder import ActionDecoder


class TestActionDecoder:
    """Test cases for ActionDecoder class."""

    def test_decoder_initialization(self):
        """Test basic decoder initialization."""
        decoder = ActionDecoder(action_format="text")
        assert decoder.action_format == "text"

    def test_decoder_with_json_format(self):
        """Test decoder with JSON format."""
        decoder = ActionDecoder(action_format="json")
        assert decoder.action_format == "json"

    def test_decode_text_format_click(self):
        """Test decoding text format action."""
        decoder = ActionDecoder(action_format="text")
        result = decoder.decode("click(100, 200)")

        assert result['type'] == 'click'
        assert result['x'] == 100
        assert result['y'] == 200

    def test_decode_text_format_type(self):
        """Test decoding text format type action."""
        decoder = ActionDecoder(action_format="text")
        result = decoder.decode("type('hello world')")

        assert result['type'] == 'type'
        assert result['text'] == 'hello world'

    def test_decode_json_format(self):
        """Test decoding JSON format action."""
        decoder = ActionDecoder(action_format="json")
        result = decoder.decode('{"type": "click", "x": 150, "y": 250}')

        assert result['type'] == 'click'
        assert result['x'] == 150
        assert result['y'] == 250

    def test_decode_invalid_text(self):
        """Test decoding invalid action text."""
        decoder = ActionDecoder(action_format="text")
        result = decoder.decode("invalid text without action format")

        # Should return error action
        assert result['type'] == 'none'
        assert 'error' in result

    def test_decode_coordinates_format(self):
        """Test decoding coordinate-based format."""
        decoder = ActionDecoder(action_format="coordinates")
        result = decoder.decode("100 200 click")

        assert result['type'] == 'click'
        assert result['x'] == 100
        assert result['y'] == 200
