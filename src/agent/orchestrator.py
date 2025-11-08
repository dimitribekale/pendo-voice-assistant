from .nlu import NLU
from .tool_manager import ToolManager
from .query_reformer import QueryReformulationTool
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self):
        self.nlu = NLU()
        self.tool_manager = ToolManager()
        self.query_reformer = QueryReformulationTool()

    def process_command(self, text):
        """
        Process a user command and execute the appropriate tool.

        Args:
            text: User's voice command as text

        Returns:
            str: Response from the tool or error message
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
                logger.info(f"Detected intent: {detected_intent}")
            except Exception as e:
                logger.error(f"NLU processing failed: {e}")
                return "I had trouble understanding that. Could you rephrase?"

            # Get the appropriate tool
            try:
                tool = self.tool_manager.get_tool(detected_intent)
            except Exception as e:
                logger.error(f"Failed to get tool for intent '{detected_intent}': {e}")
                return "I encountered an error processing your request."

            if tool:
                try:
                    # Execute the appropriate tool based on intent
                    if detected_intent == "GET_WEATHER":
                        try:
                            city = self.query_reformer.run(detected_intent, entities, text)
                            logger.info(f"Getting weather for: {city}")
                            result = tool.run(city=city)
                        except Exception as e:
                            logger.error(f"Weather query failed: {e}")
                            result = "I couldn't get the weather information right now."

                    elif detected_intent == "GET_NEWS":
                        try:
                            logger.info("Getting news headlines")
                            result = tool.run()
                        except Exception as e:
                            logger.error(f"News query failed: {e}")
                            result = "I couldn't fetch the news right now."

                    elif detected_intent == "SEARCH_WIKIPEDIA":
                        try:
                            query = self.query_reformer.run(detected_intent, entities, text)
                            logger.info(f"Searching Wikipedia for: {query}")
                            result = tool.run(query=query)
                        except Exception as e:
                            logger.error(f"Wikipedia search failed: {e}")
                            result = "I couldn't search Wikipedia right now."

                    elif detected_intent == "OPEN_CALCULATOR":
                        try:
                            logger.info("Opening calculator")
                            result = tool.run()
                        except Exception as e:
                            logger.error(f"Calculator failed: {e}")
                            result = "I couldn't open the calculator."

                    elif detected_intent == "GET_TIME":
                        try:
                            logger.info("Getting current time")
                            result = tool.run()
                        except Exception as e:
                            logger.error(f"Time query failed: {e}")
                            result = "I couldn't get the current time."

                    elif detected_intent == "TELL_JOKE":
                        try:
                            logger.info("Telling a joke")
                            result = tool.run()
                        except Exception as e:
                            logger.error(f"Joke tool failed: {e}")
                            result = "I forgot the punchline!"

                    else:
                        logger.warning(f"Unhandled intent: {detected_intent}")
                        result = "I'm not sure how to handle that request."

                    logger.info(f"Command executed successfully: {detected_intent}")
                    return result

                except Exception as e:
                    logger.exception(f"Tool execution failed for intent '{detected_intent}': {e}")
                    return "Something went wrong while processing your request."

            else:
                logger.info(f"No tool found for intent: {detected_intent}")
                return "I'm sorry, I don't understand that command."

        except Exception as e:
            # Catch-all for any unexpected errors
            logger.exception(f"Unexpected error in process_command: {e}")
            return "I encountered an unexpected error. Please try again."
