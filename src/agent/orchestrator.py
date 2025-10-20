from .nlu import NLU
from .tool_manager import ToolManager
from .query_reformer import QueryReformulationTool

class Orchestrator:
    def __init__(self):
        self.nlu = NLU()
        self.tool_manager = ToolManager()
        self.query_reformer = QueryReformulationTool()

    def process_command(self, text):
        detected_intent, entities, doc = self.nlu.process_query(text)

        tool = self.tool_manager.get_tool(detected_intent)

        if tool:
            if detected_intent == "GET_WEATHER":
                city = self.query_reformer.run(detected_intent, entities, text)
                return tool.run(city=city)

            elif detected_intent == "GET_NEWS":
                return tool.run()

            elif detected_intent == "SEARCH_WIKIPEDIA":
                query = self.query_reformer.run(detected_intent, entities, text)
                return tool.run(query=query)

            elif detected_intent == "OPEN_CALCULATOR":
                return tool.run()

            elif detected_intent == "GET_TIME":
                return tool.run()

            elif detected_intent == "TELL_JOKE":
                return tool.run()

        else:
            return "I'm sorry, I don't understand that command."
