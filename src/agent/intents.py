from enum import Enum

class Intent(str, Enum):

    GET_WEATHER = "GET_WEATHER"
    GET_NEWS = "GET_NEWS"
    SEARCH_WIKIPEDIA = "SEARCH_WIKIPEDIA"
    OPEN_CALCULATOR = "OPEN_CALCULATOR"
    GET_TIME = "GET_TIME"
    TELL_JOKE = "TELL_JOKE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, intent_str: str) -> 'Intent':
        """Convert string into Intent enum."""
        try:
            return cls(intent_str)
        except ValueError:
            cls.UNKNOWN

class IntentNames:
    """String constants for intent names"""
    GET_WEATHER = Intent.GET_WEATHER.value
    GET_NEWS = Intent.GET_NEWS.value
    SEARCH_WIKIPEDIA = Intent.SEARCH_WIKIPEDIA.value
    OPEN_CALCULATOR = Intent.OPEN_CALCULATOR.value
    GET_TIME = Intent.GET_TIME.value
    TELL_JOKE = Intent.TELL_JOKE.value
    UNKNOWN = Intent.UNKNOWN.value