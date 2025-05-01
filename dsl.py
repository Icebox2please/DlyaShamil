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
    def __init__(self, name: str, type_: DataType):
        self.name = name
        self.type = type_

    def __json__(self) -> Dict[str, Any]:
        """Сериализация параметра в JSON"""
        return {
            'name': self.name,
            'type': self.type.value
        }

class Function:
    """Функция контракта"""
    def __init__(self, name: str, params: List[Parameter], return_type: Optional[DataType], code: str):
        self.name = name
        self.params = params
        self.return_type = return_type
        self.code = code

    def __json__(self) -> Dict[str, Any]:
        """Сериализация функции в JSON"""
        return {
            'name': self.name,
            'params': [param.__json__() for param in self.params],
            'return_type': self.return_type.value,
            'code': self.code
        }

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

    def __json__(self) -> Dict[str, Any]:
        """Сериализация состояния контракта в JSON"""
        serialized_vars = {}
        for key, value in self.variables.items():
            if isinstance(value, DataType):
                serialized_vars[key] = value.value
            else:
                serialized_vars[key] = value
        return serialized_vars

class SmartContract:
    """Класс для представления смарт-контракта"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ContractState()
        self.functions = {}
        self.events = []
    
    def add_state_variable(self, name: str, type_: DataType, value: Any = None) -> None:
        """Добавление переменной состояния"""
        self.state.variables[name] = value
    
    def add_function(self, name: str, params: List[Parameter], return_type: DataType, code: str) -> None:
        """Добавление функции"""
        self.functions[name] = Function(name, params, return_type, code)
    
    def add_event(self, event: str) -> None:
        """Добавление события"""
        self.events.append(event)
    
    def __json__(self) -> Dict[str, Any]:
        """Сериализация контракта в JSON"""
        return {
            'name': self.name,
            'state': self.state.__json__(),
            'functions': {name: func.__json__() for name, func in self.functions.items()},
            'events': self.events
        }

    def __str__(self):
        return f"Contract {self.name}" 