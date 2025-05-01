from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class TokenInfo:
    """Информация о токене"""
    name: str
    symbol: str
    decimals: int
    total_supply: Decimal
    owner: str

class TokenStandard:
    """Базовый класс для стандартов токенов"""
    
    def __init__(self, state: Dict[str, Any]):
        self.state = state
        self._init_state()
    
    def _init_state(self):
        """Инициализация состояния токена"""
        if 'balances' not in self.state:
            self.state['balances'] = {}
        if 'allowances' not in self.state:
            self.state['allowances'] = {}
        if 'info' not in self.state:
            self.state['info'] = TokenInfo(
                name="",
                symbol="",
                decimals=18,
                total_supply=Decimal('0'),
                owner=""
            )
    
    def _get_balance(self, address: str) -> Decimal:
        """Получение баланса адреса"""
        return Decimal(str(self.state['balances'].get(address, '0')))
    
    def _set_balance(self, address: str, amount: Decimal):
        """Установка баланса адреса"""
        self.state['balances'][address] = str(amount)
    
    def _get_allowance(self, owner: str, spender: str) -> Decimal:
        """Получение разрешенной суммы для списания"""
        return Decimal(str(self.state['allowances'].get(f"{owner}:{spender}", '0')))
    
    def _set_allowance(self, owner: str, spender: str, amount: Decimal):
        """Установка разрешенной суммы для списания"""
        self.state['allowances'][f"{owner}:{spender}"] = str(amount)

class ERC20(TokenStandard):
    """Реализация стандарта ERC-20"""
    
    def transfer(self, to: str, amount: Decimal, sender: str) -> bool:
        """Перевод токенов"""
        if not self._is_valid_address(to):
            logger.error(f"Invalid recipient address: {to}")
            return False
        
        balance = self._get_balance(sender)
        if balance < amount:
            logger.error(f"Insufficient balance: {balance} < {amount}")
            return False
        
        self._set_balance(sender, balance - amount)
        self._set_balance(to, self._get_balance(to) + amount)
        
        logger.info(f"Transfer: {sender} -> {to}, amount: {amount}")
        return True
    
    def transfer_from(self, from_addr: str, to: str, amount: Decimal, spender: str) -> bool:
        """Перевод токенов от имени владельца"""
        if not self._is_valid_address(to):
            logger.error(f"Invalid recipient address: {to}")
            return False
        
        allowance = self._get_allowance(from_addr, spender)
        if allowance < amount:
            logger.error(f"Insufficient allowance: {allowance} < {amount}")
            return False
        
        balance = self._get_balance(from_addr)
        if balance < amount:
            logger.error(f"Insufficient balance: {balance} < {amount}")
            return False
        
        self._set_allowance(from_addr, spender, allowance - amount)
        self._set_balance(from_addr, balance - amount)
        self._set_balance(to, self._get_balance(to) + amount)
        
        logger.info(f"TransferFrom: {from_addr} -> {to}, amount: {amount}, spender: {spender}")
        return True
    
    def approve(self, spender: str, amount: Decimal, owner: str) -> bool:
        """Установка разрешения на списание"""
        if not self._is_valid_address(spender):
            logger.error(f"Invalid spender address: {spender}")
            return False
        
        self._set_allowance(owner, spender, amount)
        logger.info(f"Approve: {owner} -> {spender}, amount: {amount}")
        return True
    
    def mint(self, to: str, amount: Decimal, minter: str) -> bool:
        """Создание новых токенов"""
        if minter != self.state['info'].owner:
            logger.error(f"Only owner can mint tokens")
            return False
        
        if not self._is_valid_address(to):
            logger.error(f"Invalid recipient address: {to}")
            return False
        
        self._set_balance(to, self._get_balance(to) + amount)
        self.state['info'].total_supply += amount
        
        logger.info(f"Mint: {to}, amount: {amount}")
        return True
    
    def burn(self, amount: Decimal, burner: str) -> bool:
        """Сжигание токенов"""
        balance = self._get_balance(burner)
        if balance < amount:
            logger.error(f"Insufficient balance: {balance} < {amount}")
            return False
        
        self._set_balance(burner, balance - amount)
        self.state['info'].total_supply -= amount
        
        logger.info(f"Burn: {burner}, amount: {amount}")
        return True
    
    def _is_valid_address(self, address: str) -> bool:
        """Проверка валидности адреса"""
        return len(address) == 42 and address.startswith('0x')

class TokenFactory:
    """Фабрика для создания токенов"""
    
    @staticmethod
    def create_token(
        name: str,
        symbol: str,
        decimals: int,
        initial_supply: Decimal,
        owner: str
    ) -> ERC20:
        """Создание нового токена"""
        state = {
            'info': TokenInfo(
                name=name,
                symbol=symbol,
                decimals=decimals,
                total_supply=initial_supply,
                owner=owner
            )
        }
        
        token = ERC20(state)
        token._set_balance(owner, initial_supply)
        
        logger.info(f"Created new token: {name} ({symbol})")
        return token

# Пример использования
def example_usage():
    # Создание токена
    factory = TokenFactory()
    token = factory.create_token(
        name="MyToken",
        symbol="MTK",
        decimals=18,
        initial_supply=Decimal('1000000'),
        owner="0x1234567890123456789012345678901234567890"
    )
    
    # Тестирование функций
    alice = "0x1111111111111111111111111111111111111111"
    bob = "0x2222222222222222222222222222222222222222"
    
    # Минтинг токенов
    token.mint(alice, Decimal('100'), token.state['info'].owner)
    
    # Перевод токенов
    token.transfer(bob, Decimal('50'), alice)
    
    # Установка разрешения
    token.approve(bob, Decimal('25'), alice)
    
    # Перевод от имени
    token.transfer_from(alice, bob, Decimal('25'), bob)
    
    # Сжигание токенов
    token.burn(Decimal('25'), bob)
    
    print(f"Alice balance: {token._get_balance(alice)}")
    print(f"Bob balance: {token._get_balance(bob)}")
    print(f"Total supply: {token.state['info'].total_supply}")

if __name__ == "__main__":
    example_usage() 