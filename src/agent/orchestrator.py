from typing import Optional
from .nlu import NLU
from .tool_manager import ToolManager
from .query_reformer import QueryReformulationTool
from .intent_handlers import HandlerRegistry
from .intents import Intent
import logging


logger = logging.getLogger(__name__)

class Orchestrator:
    """Orchestrates voice command processing."""

    def __init__(
        self,
        nlu: Optional[NLU] = None,
        tool_manager: Optional[ToolManager] = None,
        query_reformer: Optional[QueryReformulationTool] = None,
        handler_registry: Optional[HandlerRegistry] = None
    ):
        self.nlu = nlu or NLU()
        self.tool_manager = tool_manager or ToolManager()
        self.query_reformer = query_reformer or QueryReformulationTool()
        self.handler_registry = handler_registry or HandlerRegistry()

        logger.info("Orchestrator initialized")

    def process_command(self, text: str) -> str:
        """
        Process a user command

        Args:
            text: User's voice command as text

        Returns:
            Response string
        """
        try:
            # Validate input
            if not text or not text.strip():
                logger.warning("Empty command received")
                return "I didn't hear anything. Please try again."

            logger.info(f"Processing command: '{text}'")

            # Process with NLU
            try:
                detected_intent, entities, doc = self.nlu.process_query(text)
                intent = Intent.from_string(detected_intent)
                logger.info(f"Detected intent: {intent}")
            except Exception as e:
                logger.error(f"NLU processing failed: {e}")
                return "I had trouble understanding that. Could you rephrase?"

            # Get the appropriate tool
            tool = self.tool_manager.get_tool(detected_intent)
            if not tool:
                logger.info(f"No tool found for intent: {intent}")
                return "I'm sorry, I don't understand that command."

            # Get the handler for this intent
            handler = self.handler_registry.get_handler(intent)
            if not handler:
                logger.warning(f"No handler registered for intent: {intent}")
                return "I'm not sure how to handle that request."

            # Execute the handler (Strategy pattern!)
            result = handler.handle(tool, entities, text, self.query_reformer)
            logger.info(f"Command executed successfully: {intent}")
            return result

        except Exception as e:
            logger.exception(f"Unexpected error in process_command: {e}")
            return "I encountered an unexpected error. Please try again."
