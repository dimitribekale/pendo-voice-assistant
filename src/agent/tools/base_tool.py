class BaseTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
