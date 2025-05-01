from typing import Dict, Any, List, Optional
import ast
from dsl import SmartContract, DataType, Parameter, Function, ContractState

class ContractParser:
    """Парсер для разбора кода смарт-контрактов"""
    
    def __init__(self):
        self.contract = None
        self.current_function = None
    
    def parse_contract(self, code: str) -> SmartContract:
        """Разбор кода контракта"""
        try:
            # Очистка кода
            code = self._clean_code(code)
            
            # Создание контракта
            self.contract = SmartContract("Contract")
            
            # Парсинг состояния
            state_vars = self._parse_state(code)
            for var_name, var_type in state_vars.items():
                self.contract.state.variables[var_name] = var_type
            
            # Парсинг функций
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._parse_function(node)
            
            return self.contract
            
        except Exception as e:
            raise ValueError(f"Error parsing contract: {str(e)}")
    
    def _clean_code(self, code: str) -> str:
        """Очистка кода от комментариев и лишних пробелов"""
        lines = []
        for line in code.split('\n'):
            # Удаление комментариев
            if '#' in line:
                line = line[:line.index('#')]
            # Удаление лишних пробелов
            line = line.strip()
            if line:
                lines.append(line)
        return '\n'.join(lines)
    
    def _parse_state(self, code: str) -> Dict[str, DataType]:
        """Разбор переменных состояния"""
        state_vars = {}
        lines = code.split('\n')
        for line in lines:
            if ':' in line and '=' not in line:
                parts = line.split(':')
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_type = parts[1].strip()
                    try:
                        state_vars[var_name] = DataType[var_type.upper()]
                    except KeyError:
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
        
        # Создание функции
        function = Function(
            name=node.name,
            params=params,
            return_type=return_type,
            code=ast.unparse(node),
            visibility='public'  # По умолчанию все функции публичные
        )
        
        # Добавление функции в контракт
        self.contract.add_function(function)
    
    def _parse_type(self, type_name: str) -> DataType:
        """Разбор типа данных"""
        try:
            return DataType[type_name.upper()]
        except KeyError:
            raise ValueError(f"Invalid type: {type_name}")

def parse_contract(code: str) -> SmartContract:
    """Вспомогательная функция для разбора контракта"""
    parser = ContractParser()
    return parser.parse_contract(code)

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