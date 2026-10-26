import abc
import numpy as np
import jinja2

'''
BasePEModule is the abstract base class for the core module for our preference elicitation task. 
Declared attributes:
'''

class BasePEModule(abc.ABC):

    def __init__(self, config, dataloader):
        
        self.config = config
        self.interactions = [] 
        self.jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=config['llm']['template_dir']))
        self.items = dataloader.get_data()

        
        self.llm = dataloader.llm
        
        self.recs = []

    '''
    Get the IDs of the top k recommended items
    '''
    def get_top_items(self, k=5):
        pass

    '''
    Generates a query based on the current utility values and the provided set of items.
    '''
    def get_query(self):
        return NotImplementedError("Abstract Base Class")
    
    '''
    Update the model's beliefs, etc based on the user's response
    '''
    def update_from_response(self,query, response):
        return NotImplementedError("Abstract Base Class")
    
    def reset(self):
        self.interactions = []
        self.recs = []
        self.total_entailment_time = 0.0
        self.total_llm_time = 0.0

    def get_last_results(self):
        return NotImplementedError
