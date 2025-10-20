from .tools.base_tool import BaseTool

class QueryReformulationTool(BaseTool):
    def __init__(self):
        super().__init__("Query Reformulation Tool", "Reformulates natural language queries into tool-specific formats.")

    def run(self, detected_intent, entities, original_text):
        # For now, this is a placeholder.
        # In Phase 2, this will be replaced by a fine-tuned LLM.
        if detected_intent == "SEARCH_WIKIPEDIA":
            query = ""
            # Attempt to extract query from original text by removing intent keywords
            if "wikipedia" in original_text:
                query_start = original_text.find("wikipedia") + len("wikipedia")
                query = original_text[query_start:].strip()
            elif "search wikipedia for" in original_text:
                query_start = original_text.find("search wikipedia for") + len("search wikipedia for")
                query = original_text[query_start:].strip()
            elif "tell me about" in original_text:
                query_start = original_text.find("tell me about") + len("tell me about")
                query = original_text[query_start:].strip()
            
            # Fallback to using entities if available and more precise
            if not query and entities:
                for ent in entities:
                    # Assuming a custom entity type 'TOPIC' or similar for Wikipedia queries
                    if ent.label_ != "INTENT": # Avoid using the intent itself as query
                        query = ent.text
                        break
            return query if query else original_text # Return original text if no specific query extracted

        # For other intents, for now, just pass through or extract primary entity
        if entities:
            for ent in entities:
                if ent.label_ == "GPE": # For weather, return city
                    return ent.text
        
        return original_text # Default: return original text
