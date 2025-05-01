from typing import Dict, Any, List, Optional
import ast
from dsl import SmartContract, DataType, Parameter, Function, ContractState

def parse_contract(code: str) -> SmartContract:
    """Вспомогательная функция для разбора контракта"""
    parser = ContractParser()
    return parser.parse_contract(code)

class ContractParser:
    """Парсер для разбора кода смарт-контрактов"""
    
    def __init__(self):
        self.contract = None
        self.current_function = None
    
    def parse_contract(self, code: str) -> SmartContract:
        """Разбор кода контракта"""
        try:
            print(f"Parsing contract code:\n{code}")  # Логируем входной код
            
            # Очистка кода
            code = self._clean_code(code)
            print(f"Cleaned code:\n{code}")  # Логируем очищенный код
            
            # Создание контракта
            self.contract = SmartContract("Contract")
            
            # Парсинг состояния
            state_vars = self._parse_state(code)
            print(f"State variables: {state_vars}")  # Логируем найденные переменные состояния
            for var_name, var_type in state_vars.items():
                self.contract.state.variables[var_name] = var_type
            
            # Парсинг функций
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._parse_function(node)
            
            return self.contract
            
        except Exception as e:
            print(f"Error details: {str(e)}")  # Логируем детали ошибки
            raise ValueError(f"Error parsing contract: {str(e)}")
    
    def _clean_code(self, code: str) -> str:
        """Очистка кода от комментариев и лишних пробелов"""
        lines = []
        for line in code.split('\n'):
            # Удаление комментариев
            if '#' in line:
                line = line[:line.index('#')]
            # Сохраняем отступы, но удаляем лишние пробелы в конце
            line = line.rstrip()
            if line:
                lines.append(line)
        return '\n'.join(lines)
    
    def _parse_state(self, code: str) -> Dict[str, DataType]:
        """Разбор переменных состояния"""
        state_vars = {}
        lines = code.split('\n')
        in_state_section = True  # Флаг для определения секции состояния
        
        for line in lines:
            # Проверяем, не началась ли секция функций
            if line.startswith('def '):
                in_state_section = False
                continue
                
            # Парсим только в секции состояния
            if in_state_section and ':' in line and '=' not in line:
                parts = line.split(':')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_type = parts[1].strip()
                    print(f"Parsing state variable: {var_name} of type {var_type}")  # Логируем каждую переменную
                    try:
                        state_vars[var_name] = DataType[var_type.upper()]
                    except KeyError:
                        print(f"Invalid type found: {var_type}")  # Логируем неверный тип
                        raise ValueError(f"Invalid type: {var_type}")
        return state_vars
    
    def _parse_function(self, node: ast.FunctionDef) -> None:
        """Разбор функции"""
        # Получение параметров
        params = []
        for arg in node.args.args:
            if arg.annotation:
                param_type = self._parse_type(arg.annotation.id)
                params.append(Parameter(arg.arg, param_type))
        
        # Получение типа возвращаемого значения
        return_type = None
        if node.returns:
            return_type = self._parse_type(node.returns.id)
        
        # Добавление функции в контракт
        self.contract.add_function(
            name=node.name,
            params=params,
            return_type=return_type,
            code=ast.unparse(node)
        )
    
    def _parse_type(self, type_name: str) -> DataType:
        """Разбор типа данных"""
        try:
            return DataType[type_name.upper()]
        except KeyError:
            raise ValueError(f"Invalid type: {type_name}")

# Пример использования
if __name__ == "__main__":
    # Пример контракта
    contract_code = """
    # Состояние контракта
    balances: MAPPING
    total_supply: INT
    
    def transfer(to: ADDRESS, amount: INT) -> BOOL:
        if balances[msg.sender] >= amount:
            balances[msg.sender] -= amount
            balances[to] += amount
            return True
        return False
    
    def balance_of(owner: ADDRESS) -> INT:
        return balances[owner]
    """
    
    # Разбор контракта
    contract = parse_contract(contract_code)
    print(f"Contract name: {contract.name}")
    print("Functions:")
    for func in contract.functions.values():
        print(f"- {func.name}") 