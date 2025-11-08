import wikipedia
import logging
from .base_tool import BaseTool

# Configure logging
logger = logging.getLogger(__name__)

class WikipediaTool(BaseTool):
    def __init__(self):
        super().__init__("Wikipedia Tool", "Searches Wikipedia for information.")

    def _search_wikipedia(self, query):
        try:
            # Get the summary of the first search result
            summary = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
            return summary
        except wikipedia.exceptions.PageError:
            return f"I couldn't find any information about {query} on Wikipedia."
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages by suggesting options
            options = e.options[:5] # Limit to 5 options
            return f"I found multiple results for {query}. Did you mean one of these?\n" + "\n".join(options)
        except Exception as e:
            return f"An unexpected error occurred while searching Wikipedia: {e}"

    def _validate_query(self, query):
        """
        Validates the Wikipedia search query.

        Args:
            query: The search query string

        Returns:
            tuple: (is_valid, error_message)
        """
        if not query:
            return False, "Please provide a search query."

        if not query.strip():
            return False, "Search query cannot be empty."

        if len(query) > 200:
            return False, "Search query is too long (max 200 characters)."

        if len(query) < 2:
            return False, "Search query is too short (min 2 characters)."

        return True, ""

    def run(self, query):
        """
        Searches Wikipedia with input validation.

        Args:
            query: The search query

        Returns:
            str: Wikipedia summary or error message
        """
        # Validate input
        is_valid, error_msg = self._validate_query(query)
        if not is_valid:
            logger.warning(f"Invalid Wikipedia query: {error_msg}")
            return error_msg

        # Sanitize query (strip whitespace)
        query = query.strip()
        logger.info(f"Searching Wikipedia for: '{query}'")

        return self._search_wikipedia(query)
