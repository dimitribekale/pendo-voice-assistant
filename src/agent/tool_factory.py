import logging
from typing import Dict, Type, Optional, List
from .tools.base_tool import BaseTool
from .tools.weather_tool import WeatherTool
from .tools.news_tool import NewsTool
from .tools.wikipedia_tool import WikipediaTool
from .tools.system_tools import CalculatorTool, TimeTool
from .tools.joke_model_tool import JokeModelTool
from .intents import Intent

logger = logging.getLogger(__name__)

class ToolFactory:
    """Factory for creating tool instances on demand."""

    def __init__(self):

        # Map intents to tool classes (not instances)
        self._tool_classes: Dict[Intent, Type[BaseTool]] = {
            Intent.GET_WEATHER: WeatherTool,
            Intent.GET_NEWS: NewsTool,
            Intent.SEARCH_WIKIPEDIA: WikipediaTool,
            Intent.OPEN_CALCULATOR: CalculatorTool,
            Intent.GET_TIME: TimeTool,
            Intent.TELL_JOKE: JokeModelTool,
        }

        # Cache for created tool instances
        self._tool_instances: Dict[Intent, BaseTool] = {}
        logger.info(f"Tool factory initialized with {len(self._tool_classes)} tool types")

    def get_tool(self, intent: Intent) -> Optional[BaseTool]:
        """
        Get a tool instance for the given intent (lazy initialization).

        Args:
            intent: The Intent enum value

        Returns:
            Tool instance, or None if intent not recognized
        """
        if intent not in self._tool_classes:
            logger.warning(f"No tool registered for intent: {intent}")
            return None
        # Return cached instance if already created
        if intent in self._tool_instances:
            logger.debug(f"Returning cached tool for intent: {intent}")
            return self._tool_instances[intent]
        
        # Create new instance
        tool_class = self._tool_classes[intent]
        try:
            logger.info(f"Creating tool instance for intent: {intent}")
            tool_instance = tool_class()
            self._tool_instances[intent] = tool_instance
            return tool_instance
        except Exception as e:
            logger.error(f"Failed to create tool for {intent}: {e}")
            return None
        
    def register_tool(self, intent: Intent, tool_class: Type[BaseTool]) -> None:
        """
        Register a new tool class for an intent.

        This allows dynamic tool registration (e.g., plugins).

        Args:
            intent: Intent to register
            tool_class: Tool class (not instance!)
        """
        logger.info(f"Registering tool {tool_class.__name__} for intent {intent}")
        self._tool_classes[intent] = tool_class

        # Invalidate cached instance if it exists
        if intent in self._tool_instances:
            del self._tool_instances[intent]
    
    def get_available_intents(self) -> List[Intent]:
        """
        Get list of all available intents.

        Returns:
            List of Intent enum values for which tools are registered
        """
        return list(self._tool_classes.keys())
