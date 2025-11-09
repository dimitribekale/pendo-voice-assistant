"""Tests for Intent enum and related functionality"""
import logging
from src.agent.intents import Intent, IntentNames

class TestIntent:

    def test_intent_values(self):
        """Test that Intent has correct string values."""
        assert Intent.GET_WEATHER.value == "GET_WEATHER"
        assert Intent.GET_NEWS.value == "GET_NEWS"
        assert Intent.SEARCH_WIKIPEDIA.value == "SEARCH_WIKIPEDIA"
        assert Intent.OPEN_CALCULATOR.value == "OPEN_CALCULATOR"
        assert Intent.GET_TIME.value == "GET_TIME"
        assert Intent.TELL_JOKE.value == "TELL_JOKE"
        assert Intent.UNKNOWN.value == "UNKNOWN"

    def test_from_string_valid_intent(self):
        """Test the validity of the string to Intent enum conversion."""
        # Arrange
        intent_string = "GET_WEATHER"
        # Act
        result = Intent.from_string(intent_string)
        # Assert
        assert result == Intent.GET_WEATHER
        assert isinstance(result, Intent)

    def test_invalid_string(self):
        """Test that invalid string returns UNKNOWN"""
        invalid_string = "INVALID_INTENT"
        result = Intent.from_string(invalid_string)
        assert result == Intent.UNKNOWN

    def test_empty_string(self):
        """Test edge case: empty string."""
        empty_string = ""
        result = Intent.from_string(empty_string)
        assert result == Intent.UNKNOWN

    def test_intent_names_match_enum(self):
        """Test that IntentNames constants match Intent enum."""
        # Assert
        assert IntentNames.GET_WEATHER == Intent.GET_WEATHER.value
        assert IntentNames.GET_NEWS == Intent.GET_NEWS.value
        assert IntentNames.UNKNOWN == Intent.UNKNOWN.value

class TestIntentEdgeCases:
    """Edge cases and defensive tests."""

    def test_from_string_lowercase(self):
        """Test that lowercase string is handled correctly."""
        result = Intent.from_string("get_weather")
        assert result == Intent.UNKNOWN  # Should be UNKNOWN because exact match fails

    def test_from_string_none(self, caplog):
        """Test handling None input logs an error and returns UNKNOWN."""
        # Arrange
        # caplog automatically captures all log messages

        # Act
        with caplog.at_level(logging.ERROR):
            result = Intent.from_string(None)

        # Assert
        assert result == Intent.UNKNOWN
        assert "received Non" in caplog.text
        assert any(record.levelname == "ERROR" for record in caplog.records)

    def test_from_string_unknown_logs_warning(self, caplog):
      """Test that unknown intent string logs a warning."""
      # Arrange
      unknown_intent = "PLAY_MUSIC"

      # Act
      with caplog.at_level(logging.WARNING):
          result = Intent.from_string(unknown_intent)

      # Assert
      assert result == Intent.UNKNOWN
      assert "Unknown intent received: 'PLAY_MUSIC'" in caplog.text
      assert any(record.levelname == "WARNING" for record in caplog.records)