import logging
from typing import Optional
from .tool_factory import ToolFactory
from .tools.base_tool import BaseTool
from .intents import Intent

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages tool retrieval using a factory pattern.

    This is a thin wrapper around ToolFactory that maintains
    backward compatibility while using lazy initialization.
    """
    def __init__(self, factory: Optional[ToolFactory] = None):
        self.factory = factory or ToolFactory()
        logger.info("ToolManager initialized")

    def get_tool(self, intent: str) -> Optional[BaseTool]:
        """Get tool for a given intent string."""
        # Convert string to Intent enum
        intent_enum = Intent.from_string(intent)
        # Get tool from factory
        return self.factory.get_tool(intent_enum)
