from abc import ABC, abstractmethod
from ..utils.setup_logging import setup_logging

class ItemScorer(ABC):
    
    def __init__(self, config):
        
        self.config = config
        self.logger = setup_logging(self.__class__.__name__, self.config)

    @abstractmethod
    def score_items(self,preference,items) -> dict:
        pass

