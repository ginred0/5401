from abc import ABC, abstractmethod

class ItemScorer(ABC):
    
    def __init__(self, config):
        
        self.config = config

    @abstractmethod
    def score_items(self,preference,items) -> dict:
        pass

