import spacy
from spacy.matcher import Matcher

class NLU:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading the spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.matcher = Matcher(self.nlp.vocab)

        # Add patterns for intent recognition
        self.matcher.add("GET_WEATHER", [[{"LOWER": "weather"}], [{"LOWER": "temperature"}]])
        self.matcher.add("GET_NEWS", [[{"LOWER": "news"}], [{"LOWER": "headlines"}]])
        self.matcher.add("SEARCH_WIKIPEDIA", [[{"LOWER": "wikipedia"}], [{"LOWER": "search"}, {"LOWER": "wikipedia"}], [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}]])
        self.matcher.add("OPEN_CALCULATOR", [[{"LOWER": "calculator"}]])
        self.matcher.add("GET_TIME", [[{"LOWER": "time"}]])
        self.matcher.add("TELL_JOKE", [[{"LOWER": "joke"}]])

    def process_query(self, text):
        doc = self.nlp(text.lower())
        matches = self.matcher(doc)

        detected_intent = None
        span = None
        entities = []

        for match_id, start, end in matches:
            detected_intent = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            break # Take the first match

        # Extract entities
        for ent in doc.ents:
            entities.append(ent)

        return detected_intent, entities, doc
