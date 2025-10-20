from .tools.weather_tool import WeatherTool
from .tools.news_tool import NewsTool
from .tools.wikipedia_tool import WikipediaTool
from .tools.system_tools import CalculatorTool, TimeTool
from .tools.joke_model_tool import JokeModelTool

class ToolManager:
    def __init__(self):
        self.tools = {
            "GET_WEATHER": WeatherTool(),
            "GET_NEWS": NewsTool(),
            "SEARCH_WIKIPEDIA": WikipediaTool(),
            "OPEN_CALCULATOR": CalculatorTool(),
            "GET_TIME": TimeTool(),
            "TELL_JOKE": JokeModelTool(),
        }

    def get_tool(self, intent):
        return self.tools.get(intent)
