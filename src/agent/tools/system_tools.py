import subprocess
import datetime
import random
import platform
import logging
from .base_tool import BaseTool



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalculatorTool(BaseTool):
    def __init__(self):
        super().__init__("Calculator Tool", "Opens the system calculator.")

    def run(self):
        """
        Opens the system's calculator application.
        Supports Windows, macOS, and Linux (GNOME).
        """
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(
                    ["calc.exe"],
                    check=True,
                    timeout=5
                )
            elif system == "Darwin":
                subprocess.run(
                    ["open", "-a", "Calculator"],
                    check=True,
                    timeout=5
                )
            elif system == "Linux":
                subprocess.run(
                    ["gnome-calculator"],
                    check=True,
                    timeout=5
                )
            else:
                logger.warning(f"Calculator not supported on {system}")
                return f"Sorry, calculator is not supported on {system}."
            return "Opening calculator."
        
        except subprocess.TimeoutExpired:
            logger.error("Calculator failed to start (timeout)")
            return "Calculator took too long to start."
        except FileNotFoundError:
            logger.error(f"Calculator application not found on {system}")
            return "Calculator application not found on your system."
        except subprocess.CalledProcessError as e:
            logger.error(f"Calculator failed to start: {e}")
            return "Failed to open calculator."
        except Exception as e:
            logger.exception(f"Unexpected error opening calculator: {e}")
            return "An unexpected error occured while opening calculator."



class TimeTool(BaseTool):
    def __init__(self):
        super().__init__("Time Tool", "Tells the current time.")

    def run(self):
        """Returns the current time in a human-readable format."""
        now = datetime.datetime.now()
        time_str = now.strftime('%I:%M %p')
        return f"The current time is {time_str}."



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
