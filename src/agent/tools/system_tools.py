import subprocess
import datetime
import random
from .base_tool import BaseTool

class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("Calculator Tool", "Opens the system calculator.")

    def run(self):
        subprocess.run(["calc.exe"])
        return "Opening calculator."

class TimeTool(BaseTool):
    def __init__(self):
        super().__init__("Time Tool", "Tells the current time.")

    def run(self):
        now = datetime.datetime.now()
        return f"The current time is {now.hour} {now.minute}."

class JokeTool(BaseTool):
    def __init__(self):
        super().__init__("Joke Tool", "Tells a random joke.")

    def run(self):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An Impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return random.choice(jokes)
