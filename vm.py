from typing import Dict, Any, List, Optional, Union
import logging
import ast
import sys
import re
import time
import math
import datetime
import hashlib
import base58
import ecdsa
import binascii
from decimal import Decimal, getcontext, ROUND_FLOOR, ROUND_CEILING
from dsl import SmartContract, ContractState, DataType, Parameter
from standards import ERC20  # Добавляем импорт ERC20

logger = logging.getLogger(__name__)

# Настройка точности для десятичных вычислений
getcontext().prec = 28

# Криптографические константы
CURVE = ecdsa.SECP256k1
HASH_ALGORITHM = hashlib.sha256
ADDRESS_PREFIX = "0x"
ADDRESS_LENGTH = 42  # 0x + 40 hex chars
PRIVATE_KEY_LENGTH = 64  # 32 bytes in hex
PUBLIC_KEY_LENGTH = 130  # 65 bytes in hex (uncompressed)
SIGNATURE_LENGTH = 128  # 64 bytes in hex

class VMError(Exception):
    """Базовый класс для ошибок виртуальной машины"""
    def __init__(self, message: str, operation: str = None, gas_used: int = None):
        self.message = message
        self.operation = operation
        self.gas_used = gas_used
        self.timestamp = datetime.datetime.now()
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Форматирование сообщения об ошибке"""
        parts = [self.message]
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        if self.gas_used is not None:
            parts.append(f"Gas used: {self.gas_used}")
        parts.append(f"Time: {self.timestamp}")
        return " | ".join(parts)

class GasLimitExceeded(VMError):
    """Ошибка превышения лимита газа"""
    def __init__(self, message: str, gas_used: int, gas_limit: int, operation: str = None):
        self.gas_limit = gas_limit
        super().__init__(
            f"{message} (Used: {gas_used}, Limit: {gas_limit})",
            operation=operation,
            gas_used=gas_used
        )

class InvalidOperation(VMError):
    """Ошибка недопустимой операции"""
    def __init__(self, operation: str, reason: str, gas_used: int = None):
        super().__init__(
            f"Invalid operation: {operation} - {reason}",
            operation=operation,
            gas_used=gas_used
        )

class StackOverflow(VMError):
    """Ошибка переполнения стека"""
    def __init__(self, current_depth: int, max_depth: int, operation: str = None):
        super().__init__(
            f"Stack overflow: current depth {current_depth} exceeds maximum {max_depth}",
            operation=operation
        )

class StringOperationError(VMError):
    """Ошибка при выполнении строковой операции"""
    def __init__(self, operation: str, string: str, max_length: int = None, reason: str = None):
        message = f"String operation error: {operation}"
        if max_length and len(string) > max_length:
            message += f" - String length {len(string)} exceeds maximum {max_length}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, operation=operation)

class MathError(VMError):
    """Ошибка при выполнении математической операции"""
    def __init__(self, operation: str, operands: List[Any], reason: str):
        super().__init__(
            f"Math error in {operation} with operands {operands}: {reason}",
            operation=operation
        )

class DateTimeError(VMError):
    """Ошибка при работе с датой и временем"""
    def __init__(self, operation: str, datetime_value: Any, reason: str):
        super().__init__(
            f"DateTime error in {operation} with value {datetime_value}: {reason}",
            operation=operation
        )

class CryptoError(VMError):
    """Ошибка при выполнении криптографической операции"""
    def __init__(self, operation: str, data: Any = None, reason: str = None):
        message = f"Crypto error in {operation}"
        if data:
            message += f" with data: {data}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, operation=operation)

class AddressError(VMError):
    """Ошибка при работе с адресом"""
    def __init__(self, operation: str, address: str, reason: str = None):
        message = f"Address error in {operation} for address {address}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, operation=operation)

class GasPricing:
    """Константы для ценообразования газа"""
    # Базовые операции
    BASE_OPERATION = 2
    STORAGE_READ = 200
    STORAGE_WRITE = 5000
    STORAGE_DELETE = 5000
    
    # Математические операции
    ADDITION = 3
    SUBTRACTION = 3
    MULTIPLICATION = 5
    DIVISION = 5
    MODULO = 5
    
    # Строковые операции
    STRING_CONCAT = 50
    STRING_LENGTH = 3
    STRING_SLICE = 50
    
    # Криптографические операции
    HASH_OPERATION = 60
    SIGNATURE_VERIFICATION = 3000
    KEY_GENERATION = 65000
    
    # Токенные операции
    TOKEN_TRANSFER = 21000
    TOKEN_APPROVE = 46000
    TOKEN_MINT = 50000
    TOKEN_BURN = 30000
    
    # Контрактные операции
    CONTRACT_CREATION = 53000
    CONTRACT_CALL = 21000
    EVENT_EMISSION = 1000
    
    # Сложные операции
    LOOP_ITERATION = 3
    CONDITIONAL_CHECK = 3
    FUNCTION_CALL = 700
    MEMORY_ALLOCATION = 3

class VirtualMachine:
    """Виртуальная машина для выполнения смарт-контрактов"""
    
    def __init__(self):
        self.state = None
        self.caller = None
    
    def execute(self, contract: SmartContract, function_name: str, args: List[Any], caller: str) -> Any:
        """Выполнение функции контракта"""
        try:
            # Инициализация состояния
            self.state = contract.state
            self.caller = caller
            
            # Получение функции
            if function_name not in contract.functions:
                raise ValueError(f"Function {function_name} not found")
            
            function = contract.functions[function_name]
            
            # Проверка аргументов
            if len(args) != len(function.params):
                raise ValueError(f"Expected {len(function.params)} arguments, got {len(args)}")
            
            # Подготовка локального контекста
            local_vars = {
                'state': self.state,
                'caller': self.caller,
                'args': args
            }
            
            # Добавление аргументов в локальный контекст
            for param, arg in zip(function.params, args):
                local_vars[param.name] = arg
            
            # Парсинг и выполнение кода
            try:
                tree = ast.parse(function.code)
                result = self._execute_ast(tree, local_vars)
                return result
            except Exception as e:
                raise VMError(f"Error executing function: {str(e)}")
                
        finally:
            # Очистка состояния
            self.state = None
            self.caller = None
    
    def _execute_ast(self, node: ast.AST, local_vars: Dict[str, Any]) -> Any:
        """Выполнение AST узла"""
        if isinstance(node, ast.Module):
            return self._execute_module(node, local_vars)
        elif isinstance(node, ast.Expr):
            return self._execute_expr(node, local_vars)
        elif isinstance(node, ast.Assign):
            return self._execute_assign(node, local_vars)
        elif isinstance(node, ast.Return):
            return self._execute_return(node, local_vars)
        elif isinstance(node, ast.Call):
            return self._execute_call(node, local_vars)
        elif isinstance(node, ast.Name):
            return self._execute_name(node, local_vars)
        elif isinstance(node, ast.Attribute):
            return self._execute_attribute(node, local_vars)
        elif isinstance(node, ast.BinOp):
            return self._execute_binop(node, local_vars)
        elif isinstance(node, ast.Compare):
            return self._execute_compare(node, local_vars)
        elif isinstance(node, ast.If):
            return self._execute_if(node, local_vars)
        elif isinstance(node, ast.For):
            return self._execute_for(node, local_vars)
        elif isinstance(node, ast.While):
            return self._execute_while(node, local_vars)
        elif isinstance(node, ast.Break):
            return self._execute_break()
        elif isinstance(node, ast.Continue):
            return self._execute_continue()
        elif isinstance(node, ast.List):
            return self._execute_list(node, local_vars)
        elif isinstance(node, ast.Subscript):
            return self._execute_subscript(node, local_vars)
        elif isinstance(node, ast.FunctionDef):
            # Для определения функции просто пропускаем его, так как функции уже определены в контракте
            return None
        else:
            raise VMError(f"Unsupported AST node: {type(node)}")
    
    def _execute_module(self, node: ast.Module, local_vars: Dict[str, Any]) -> Any:
        """Выполнение модуля"""
        result = None
        for stmt in node.body:
            result = self._execute_ast(stmt, local_vars)
        return result
    
    def _execute_expr(self, node: ast.Expr, local_vars: Dict[str, Any]) -> Any:
        """Выполнение выражения"""
        return self._execute_ast(node.value, local_vars)
    
    def _execute_assign(self, node: ast.Assign, local_vars: Dict[str, Any]) -> None:
        """Выполнение присваивания"""
        value = self._execute_ast(node.value, local_vars)
        for target in node.targets:
            if isinstance(target, ast.Name):
                local_vars[target.id] = value
            elif isinstance(target, ast.Attribute):
                obj = self._execute_ast(target.value, local_vars)
                setattr(obj, target.attr, value)
            elif isinstance(target, ast.Subscript):
                obj = self._execute_ast(target.value, local_vars)
                key = self._execute_ast(target.slice, local_vars)
                obj[key] = value
            else:
                raise VMError(f"Unsupported assignment target: {type(target)}")
    
    def _execute_return(self, node: ast.Return, local_vars: Dict[str, Any]) -> Any:
        """Выполнение return"""
        if node.value is None:
            return None
        return self._execute_ast(node.value, local_vars)
    
    def _execute_call(self, node: ast.Call, local_vars: Dict[str, Any]) -> Any:
        """Выполнение вызова функции"""
        func = self._execute_ast(node.func, local_vars)
        args = [self._execute_ast(arg, local_vars) for arg in node.args]
        
        if callable(func):
            return func(*args)
        
        raise VMError(f"Object is not callable: {type(func)}")
    
    def _execute_name(self, node: ast.Name, local_vars: Dict[str, Any]) -> Any:
        """Выполнение обращения к имени"""
        if node.id in local_vars:
            return local_vars[node.id]
        raise VMError(f"Name not found: {node.id}")
    
    def _execute_attribute(self, node: ast.Attribute, local_vars: Dict[str, Any]) -> Any:
        """Выполнение обращения к атрибуту"""
        obj = self._execute_ast(node.value, local_vars)
        return getattr(obj, node.attr)
    
    def _execute_binop(self, node: ast.BinOp, local_vars: Dict[str, Any]) -> Any:
        """Выполнение бинарной операции"""
        left = self._execute_ast(node.left, local_vars)
        right = self._execute_ast(node.right, local_vars)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Mod):
            return left % right
        else:
            raise VMError(f"Unsupported binary operation: {type(node.op)}")
    
    def _execute_compare(self, node: ast.Compare, local_vars: Dict[str, Any]) -> bool:
        """Выполнение операции сравнения"""
        left = self._execute_ast(node.left, local_vars)
        for op, right in zip(node.ops, node.comparators):
            right_val = self._execute_ast(right, local_vars)
            
            if isinstance(op, ast.Eq):
                if left != right_val:
                    return False
            elif isinstance(op, ast.NotEq):
                if left == right_val:
                    return False
            elif isinstance(op, ast.Lt):
                if left >= right_val:
                    return False
            elif isinstance(op, ast.LtE):
                if left > right_val:
                    return False
            elif isinstance(op, ast.Gt):
                if left <= right_val:
                    return False
            elif isinstance(op, ast.GtE):
                if left < right_val:
                    return False
            else:
                raise VMError(f"Unsupported comparison operation: {type(op)}")
            
            left = right_val
        
        return True
    
    def _execute_if(self, node: ast.If, local_vars: Dict[str, Any]) -> Any:
        """Выполнение условного оператора"""
        test = self._execute_ast(node.test, local_vars)
        if test:
            for stmt in node.body:
                result = self._execute_ast(stmt, local_vars)
                if isinstance(result, (ast.Break, ast.Continue)):
                    return result
        else:
            for stmt in node.orelse:
                result = self._execute_ast(stmt, local_vars)
                if isinstance(result, (ast.Break, ast.Continue)):
                    return result
        return None
    
    def _execute_for(self, node: ast.For, local_vars: Dict[str, Any]) -> Any:
        """Выполнение цикла for"""
        iterable = self._execute_ast(node.iter, local_vars)
        for item in iterable:
            if isinstance(node.target, ast.Name):
                local_vars[node.target.id] = item
            else:
                raise VMError("Only simple variable names are supported in for loops")
            
            for stmt in node.body:
                result = self._execute_ast(stmt, local_vars)
                if isinstance(result, ast.Break):
                    return None
                elif isinstance(result, ast.Continue):
                    break
        return None
    
    def _execute_while(self, node: ast.While, local_vars: Dict[str, Any]) -> Any:
        """Выполнение цикла while"""
        while self._execute_ast(node.test, local_vars):
            for stmt in node.body:
                result = self._execute_ast(stmt, local_vars)
                if isinstance(result, ast.Break):
                    return None
                elif isinstance(result, ast.Continue):
                    break
        return None
    
    def _execute_break(self) -> ast.Break:
        """Выполнение break"""
        return ast.Break()
    
    def _execute_continue(self) -> ast.Continue:
        """Выполнение continue"""
        return ast.Continue()
    
    def _execute_list(self, node: ast.List, local_vars: Dict[str, Any]) -> List[Any]:
        """Выполнение создания списка"""
        return [self._execute_ast(elt, local_vars) for elt in node.elts]
    
    def _execute_subscript(self, node: ast.Subscript, local_vars: Dict[str, Any]) -> Any:
        """Выполнение обращения по индексу"""
        value = self._execute_ast(node.value, local_vars)
        key = self._execute_ast(node.slice, local_vars)
        return value[key]

# Пример использования
def example_usage():
    from dsl import ContractDSL, ContractType, Parameter, DataType
    
    # Создание DSL и контракта токена
    dsl = ContractDSL()
    token_contract = dsl.create_token_contract(
        name="MyToken",
        symbol="MTK",
        decimals=18,
        initial_supply=Decimal('1000000'),
        owner="0x1234567890123456789012345678901234567890"
    )
    
    # Создание и выполнение виртуальной машины
    vm = VirtualMachine()
    
    # Тестирование операций с токенами
    alice = "0x1111111111111111111111111111111111111111"
    bob = "0x2222222222222222222222222222222222222222"
    
    # Перевод токенов
    result1 = vm.execute(token_contract, "transfer", [bob, Decimal('100')], alice)
    print(f"Transfer result: {result1}")
    
    # Проверка баланса
    balance = vm._safe_token_balance("MTK_0", bob)
    print(f"Bob's balance: {balance}")
    
    # Установка разрешения
    result2 = vm.execute(token_contract, "approve", [bob, Decimal('50')], alice)
    print(f"Approve result: {result2}")
    
    # Проверка разрешения
    allowance = vm._safe_token_allowance("MTK_0", alice, bob)
    print(f"Bob's allowance from Alice: {allowance}")
    
    print(f"State: {token_contract.state}")
    print(f"Gas used: {vm.gas_used}")

if __name__ == "__main__":
    example_usage() 