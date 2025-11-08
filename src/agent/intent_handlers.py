import logging
from abc import ABC, abstractmethod
from .intents import Intent
from typing import Optional
from .tools.base_tool import BaseTool
from .query_reformer import QueryReformulationTool

logger = logging.getLogger(__name__)

class IntentHandler(ABC):
    """
    Abstract base class for intent handlers.
    Each concrete handler implements how to execute its specific intent.
    """
    @abstractmethod
    def handle(
        self,
        tool: BaseTool,
        entities: list,
        text: str,
        query_reformer: Optional[QueryReformulationTool] = None
    ) -> str:
        """
        Handle the intent execution.

        Args:
            tool: The tool instance to execute
            entities: Entities extracted by NLU
            text: Original user text
            query_reformer: Query reformulation tool (optional)

        Returns:
            Response string
        """
        pass

class WeatherHandler(IntentHandler):
    
    def handle(self, tool, entities, text, query_reformer=None) -> str:
        try:
            city = None
            if query_reformer:
                city = query_reformer.run("GET_WEATHER", entities, text)
            logger.info(f"Getting weather for: {city}")
            return tool.run(city=city)
        except Exception as e:
            logger.error(f"Weather query failed: {e}")
            return "I couldn't get the weather information right now."

class NewsHandler(IntentHandler):
    """Handler for GET_NEWS intent."""

    def handle(self, tool, entities, text, query_reformer=None) -> str:
        try:
            logger.info("Getting news headlines")
            return tool.run()
        except Exception as e:
            logger.error(f"News query failed: {e}")
            return "I couldn't fetch the news right now."

class WikipediaHandler(IntentHandler):

    def handle(self, tool, entities, text, query_reformer=None) -> str:
        try:
            query = None
            if query_reformer:
                query = query_reformer.run("SEARCH_WIKIPEDIA", entities, text)
            logger.info(f"Searching Wikipedia for: {query}")
            return tool.run(query=query)
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return "I couldn't search Wikipedia right now."
        
class SimpleHandler(IntentHandler):
    """
    Handler for simple intents that don't need query reformulation.

    Works for: OPEN_CALCULATOR, GET_TIME, TELL_JOKE
    """
    def __init__(self, intent_name: str, error_message: str):
        self.intent_name = intent_name
        self.error_message = error_message

    def handle(self, tool, entities, text, query_reformer=None) -> str:
        try:
            logger.info(f"Executing: {self.intent_name}")
            return tool.run()
        except Exception as e:
            logger.error(f"{self.intent_name} failed: {e}")
            return self.error_message
        
class HandlerRegistry:
    """Registery mapping intents to their handlers."""

    def __inti__(self):

        self._handlers = {
            Intent.GET_WEATHER: WeatherHandler(),
            Intent.GET_NEWS: NewsHandler(),
            Intent.SEARCH_WIKIPEDIA: WikipediaHandler(),
            Intent.OPEN_CALCULATOR: SimpleHandler("Open Calculator", "I couldn't open the calculator."),
            Intent.GET_TIME: SimpleHandler("Get Time", "I couldn't get the current time."),
            Intent.TELL_JOKE: SimpleHandler("Tell Joke", "I forgot the punchline!"),
        }
    def get_handler(self, intent) -> Optional[IntentHandler]:
        return self._handlers.get(intent)