import requests
from src.config import Config
from .base_tool import BaseTool

class NewsTool(BaseTool):
    def __init__(self):
        super().__init__("News Tool", "Provides top news headlines.")
        self.api_key = Config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/top-headlines"

    def _get_top_headlines(self, country="us"):
        if not self.api_key:
            return "API key for NewsAPI is not provided. Please get a key from https://newsapi.org/"

        params = {
            "country": country,
            "apiKey": self.api_key,
            "pageSize": 5  # Limit to 5 headlines
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            news_data = response.json()

            if news_data["status"] != "ok":
                return f"Could not get news headlines. Error: {news_data.get('message', 'Unknown error')}"

            articles = news_data["articles"]
            headlines = [article["title"] for article in articles]

            if not headlines:
                return "I couldn't find any top headlines right now."

            return "Here are the top 5 headlines:\n" + "\n".join(f"- {headline}" for headline in headlines)

        except requests.exceptions.RequestException as e:
            return f"Could not get news headlines. Error: {e}"

    def run(self, country="us"):
        return self._get_top_headlines(country)
