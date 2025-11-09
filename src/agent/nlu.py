import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from typing import Optional, List, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)

class NLU:
    """
    Natural Language Understanding component.

    Uses spaCy for text processing and pattern matching to detect
    user intents and extract entities from voice commands.
    """
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.info("SpaCy model not found. Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model downloaded and loaded successfully")

        self.matcher = Matcher(self.nlp.vocab)

        # Add patterns for intent recognition
        self.matcher.add("GET_WEATHER", [[{"LOWER": "weather"}], [{"LOWER": "temperature"}]])
        self.matcher.add("GET_NEWS", [[{"LOWER": "news"}], [{"LOWER": "headlines"}]])
        self.matcher.add("SEARCH_WIKIPEDIA", [[{"LOWER": "wikipedia"}], [{"LOWER": "search"}, {"LOWER": "wikipedia"}], [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}]])
        self.matcher.add("OPEN_CALCULATOR", [[{"LOWER": "calculator"}]])
        self.matcher.add("GET_TIME", [[{"LOWER": "time"}]])
        self.matcher.add("TELL_JOKE", [[{"LOWER": "joke"}]])

    def process_query(self, text: str) -> Tuple[Optional[str], List[Any], Doc]:
        """
        Process user query to detect intent and extract entities.

        Args:
            text: User's voice command as text

        Returns:
            Tuple containing:
                - detected_intent: Intent string or None if not detected
                - entities: List of spaCy entities found in text
                - doc: Processed spaCy Doc object
        """
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
