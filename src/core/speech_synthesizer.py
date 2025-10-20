import pyttsx3

class SpeechSynthesizer:
    def __init__(self):
        self.engine = pyttsx3.init()

    def say(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
