from flask.json import JSONEncoder
from dsl import SmartContract, Function, Parameter, ContractState

class CustomJSONEncoder(JSONEncoder):
    """Кастомный JSON-кодировщик для наших классов"""
    
    def default(self, obj):
        if hasattr(obj, '__json__'):
            return obj.__json__()
        return super().default(obj) 