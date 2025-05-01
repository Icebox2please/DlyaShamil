from typing import Dict, Any, List, Optional, Union
from enum import Enum
import time

class DataType(Enum):
    """Типы данных в контракте"""
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    ADDRESS = "address"
    MAPPING = "mapping"

class Parameter:
    """Параметр функции"""
    def __init__(self, name: str, type: DataType):
        self.name = name
        self.type = type

class Function:
    """Функция контракта"""
    def __init__(self, name: str, params: List[Parameter], return_type: Optional[DataType], code: str):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.code = code

class ContractState:
    """Состояние контракта"""
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.events: List[Dict] = []

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any):
        self.variables[key] = value

    def add_event(self, event: Dict[str, Any]):
        self.events.append(event)

class SmartContract:
    """Базовый класс смарт-контракта"""
    def __init__(self, name: str):
        self.name = name
        self.state = ContractState()
        self.functions: Dict[str, Function] = {}

    def add_function(self, name: str, params: List[Parameter], return_type: Optional[DataType], code: str):
        """Добавление функции в контракт"""
        if name in self.functions:
            raise ValueError(f"Function {name} already exists")
        self.functions[name] = Function(name, params, return_type, code)

    def __str__(self):
        return f"Contract {self.name}" 