import wikipedia
from .base_tool import BaseTool

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

    def run(self, query):
        return self._search_wikipedia(query)
