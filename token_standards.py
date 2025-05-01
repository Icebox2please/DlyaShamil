from typing import Dict, Any, List, Optional, Union, Set
from decimal import Decimal
from dataclasses import dataclass
import logging
from enum import Enum
import time
import json
from datetime import datetime
from token_validator import TokenValidator, ValidationError

logger = logging.getLogger(__name__)

class TokenType(Enum):
    """Типы токенов"""
    ERC20 = "ERC20"
    ERC721 = "ERC721"
    ERC1155 = "ERC1155"

class TokenRole(Enum):
    """Роли токенов"""
    ADMIN = "admin"
    MINTER = "minter"
    OPERATOR = "operator"
    VIEWER = "viewer"

@dataclass
class TokenPermissions:
    """Разрешения токена"""
    can_mint: bool = False
    can_burn: bool = False
    can_transfer: bool = False
    can_approve: bool = False
    can_create_collection: bool = False
    can_manage_collection: bool = False
    can_view: bool = True

@dataclass
class TokenMetadata:
    """Метаданные токена"""
    name: str
    description: str
    image: str
    attributes: Dict[str, Any]
    created_at: float = time.time()
    updated_at: float = time.time()
    version: str = "1.0.0"
    
    def to_json(self) -> str:
        """Преобразование метаданных в JSON"""
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "image": self.image,
            "attributes": self.attributes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TokenMetadata':
        """Создание метаданных из JSON"""
        errors = TokenValidator.validate_metadata_json(json_str)
        if errors:
            raise ValueError(f"Invalid metadata: {errors}")
        data = json.loads(json_str)
        return cls(**data)
    
    def update(self, **kwargs) -> None:
        """Обновление метаданных"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = time.time()
        
        # Валидация обновленных метаданных
        errors = TokenValidator.validate_metadata(self.__dict__)
        if errors:
            raise ValueError(f"Invalid metadata after update: {errors}")

@dataclass
class TokenEvent:
    """Событие токена"""
    event_type: str
    token_id: int
    from_addr: Optional[str]
    to_addr: Optional[str]
    amount: Optional[Decimal]
    metadata: Optional[TokenMetadata]
    timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование события в словарь"""
        return {
            "event_type": self.event_type,
            "token_id": self.token_id,
            "from_addr": self.from_addr,
            "to_addr": self.to_addr,
            "amount": str(self.amount) if self.amount else None,
            "metadata": self.metadata.to_json() if self.metadata else None,
            "timestamp": self.timestamp
        }

@dataclass
class NFT:
    """NFT токен"""
    token_id: int
    owner: str
    metadata: TokenMetadata
    created_at: float
    events: List[TokenEvent] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
    
    def add_event(self, event: TokenEvent) -> None:
        """Добавление события"""
        self.events.append(event)
    
    def get_events(self, event_type: Optional[str] = None) -> List[TokenEvent]:
        """Получение событий"""
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events

@dataclass
class MultiToken:
    """Multi-Token (ERC-1155)"""
    token_id: int
    token_type: int
    supply: Decimal
    metadata: TokenMetadata
    owners: Dict[str, Decimal]  # address -> balance
    events: List[TokenEvent] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = []
    
    def add_event(self, event: TokenEvent) -> None:
        """Добавление события"""
        self.events.append(event)
    
    def get_events(self, event_type: Optional[str] = None) -> List[TokenEvent]:
        """Получение событий"""
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events

@dataclass
class TokenCollection:
    """Коллекция токенов"""
    name: str
    description: str
    tokens: List[int]
    created_at: float = time.time()
    updated_at: float = time.time()
    
    def add_token(self, token_id: int) -> bool:
        """Добавление токена в коллекцию"""
        if not TokenValidator.validate_token_id(token_id):
            logger.error(f"Invalid token ID: {token_id}")
            return False
        
        if token_id in self.tokens:
            logger.error(f"Token {token_id} already in collection")
            return False
        
        self.tokens.append(token_id)
        self.updated_at = time.time()
        return True
    
    def remove_token(self, token_id: int) -> bool:
        """Удаление токена из коллекции"""
        if token_id not in self.tokens:
            logger.error(f"Token {token_id} not in collection")
            return False
        
        self.tokens.remove(token_id)
        self.updated_at = time.time()
        return True
    
    def has_token(self, token_id: int) -> bool:
        """Проверка наличия токена в коллекции"""
        return token_id in self.tokens
    
    def get_tokens(self) -> List[int]:
        """Получение списка токенов"""
        return self.tokens.copy()

class TokenAccessControl:
    """Управление доступом к токенам"""
    
    def __init__(self):
        self.roles: Dict[str, Set[TokenRole]] = {}  # address -> roles
        self.role_permissions: Dict[TokenRole, TokenPermissions] = {
            TokenRole.ADMIN: TokenPermissions(
                can_mint=True,
                can_burn=True,
                can_transfer=True,
                can_approve=True,
                can_create_collection=True,
                can_manage_collection=True,
                can_view=True
            ),
            TokenRole.MINTER: TokenPermissions(
                can_mint=True,
                can_view=True
            ),
            TokenRole.OPERATOR: TokenPermissions(
                can_transfer=True,
                can_approve=True,
                can_view=True
            ),
            TokenRole.VIEWER: TokenPermissions(
                can_view=True
            )
        }
    
    def grant_role(self, address: str, role: TokenRole) -> bool:
        """Назначение роли"""
        if not TokenValidator.validate_address(address):
            logger.error(f"Invalid address: {address}")
            return False
        
        if address not in self.roles:
            self.roles[address] = set()
        self.roles[address].add(role)
        logger.info(f"Granted role {role.value} to {address}")
        return True
    
    def revoke_role(self, address: str, role: TokenRole) -> bool:
        """Отзыв роли"""
        if address not in self.roles or role not in self.roles[address]:
            logger.error(f"Address {address} does not have role {role.value}")
            return False
        
        self.roles[address].remove(role)
        if not self.roles[address]:
            del self.roles[address]
        logger.info(f"Revoked role {role.value} from {address}")
        return True
    
    def has_role(self, address: str, role: TokenRole) -> bool:
        """Проверка наличия роли"""
        return address in self.roles and role in self.roles[address]
    
    def get_roles(self, address: str) -> Set[TokenRole]:
        """Получение ролей адреса"""
        return self.roles.get(address, set())
    
    def check_permission(self, address: str, permission: str) -> bool:
        """Проверка разрешения"""
        if address not in self.roles:
            return False
        
        for role in self.roles[address]:
            permissions = self.role_permissions[role]
            if hasattr(permissions, permission) and getattr(permissions, permission):
                return True
        
        return False

class ERC721:
    """Реализация стандарта ERC-721 (NFT)"""
    
    def __init__(self, name: str, symbol: str):
        self.name = name
        self.symbol = symbol
        self.tokens: Dict[int, NFT] = {}
        self.owner_tokens: Dict[str, List[int]] = {}  # owner -> token_ids
        self.approvals: Dict[int, str] = {}  # token_id -> approved_address
        self.operators: Dict[str, List[str]] = {}  # owner -> operator_addresses
        self.events: List[TokenEvent] = []
        self.collections: Dict[str, TokenCollection] = {}
        self.access_control = TokenAccessControl()
    
    def mint(self, to: str, token_id: int, metadata: TokenMetadata, caller: str) -> bool:
        """Создание нового NFT"""
        if not self.access_control.check_permission(caller, "can_mint"):
            logger.error(f"Caller {caller} does not have permission to mint")
            return False
        
        if not TokenValidator.validate_address(to):
            logger.error(f"Invalid address: {to}")
            return False
        
        if not TokenValidator.validate_token_id(token_id):
            logger.error(f"Invalid token ID: {token_id}")
            return False
        
        if token_id in self.tokens:
            logger.error(f"Token {token_id} already exists")
            return False
        
        self.tokens[token_id] = NFT(
            token_id=token_id,
            owner=to,
            metadata=metadata,
            created_at=time.time()
        )
        
        if to not in self.owner_tokens:
            self.owner_tokens[to] = []
        self.owner_tokens[to].append(token_id)
        
        # Добавляем событие
        event = TokenEvent(
            event_type="Mint",
            token_id=token_id,
            from_addr=None,
            to_addr=to,
            amount=None,
            metadata=metadata
        )
        self.tokens[token_id].add_event(event)
        self.events.append(event)
        
        logger.info(f"Minted NFT {token_id} to {to}")
        return True
    
    def create_collection(self, name: str, description: str, caller: str) -> str:
        """Создание коллекции токенов"""
        if not self.access_control.check_permission(caller, "can_create_collection"):
            logger.error(f"Caller {caller} does not have permission to create collections")
            return ""
        
        if name in self.collections:
            logger.error(f"Collection {name} already exists")
            return ""
        
        self.collections[name] = TokenCollection(
            name=name,
            description=description,
            tokens=[]
        )
        return name
    
    def add_to_collection(self, collection_name: str, token_id: int, caller: str) -> bool:
        """Добавление токена в коллекцию"""
        if not self.access_control.check_permission(caller, "can_manage_collection"):
            logger.error(f"Caller {caller} does not have permission to manage collections")
            return False
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return False
        
        return self.collections[collection_name].add_token(token_id)
    
    def remove_from_collection(self, collection_name: str, token_id: int) -> bool:
        """Удаление токена из коллекции"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return False
        
        return self.collections[collection_name].remove_token(token_id)
    
    def get_collection_tokens(self, collection_name: str) -> List[int]:
        """Получение токенов из коллекции"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return []
        
        return self.collections[collection_name].get_tokens()
    
    def get_collections(self) -> Dict[str, TokenCollection]:
        """Получение всех коллекций"""
        return self.collections.copy()
    
    def transfer(self, from_addr: str, to: str, token_id: int, caller: str) -> bool:
        """Передача NFT"""
        if not self.access_control.check_permission(caller, "can_transfer"):
            logger.error(f"Caller {caller} does not have permission to transfer")
            return False
        
        if token_id not in self.tokens:
            logger.error(f"Token {token_id} does not exist")
            return False
        
        token = self.tokens[token_id]
        if token.owner != from_addr:
            logger.error(f"Token {token_id} is not owned by {from_addr}")
            return False
        
        # Обновляем владельца
        token.owner = to
        self.owner_tokens[from_addr].remove(token_id)
        
        if to not in self.owner_tokens:
            self.owner_tokens[to] = []
        self.owner_tokens[to].append(token_id)
        
        # Очищаем одобрения
        if token_id in self.approvals:
            del self.approvals[token_id]
        
        # Добавляем событие
        event = TokenEvent(
            event_type="Transfer",
            token_id=token_id,
            from_addr=from_addr,
            to_addr=to,
            amount=None,
            metadata=token.metadata
        )
        token.add_event(event)
        self.events.append(event)
        
        logger.info(f"Transferred NFT {token_id} from {from_addr} to {to}")
        return True
    
    def approve(self, approved: str, token_id: int, owner: str) -> bool:
        """Одобрение передачи NFT"""
        if token_id not in self.tokens:
            logger.error(f"Token {token_id} does not exist")
            return False
        
        token = self.tokens[token_id]
        if token.owner != owner:
            logger.error(f"Token {token_id} is not owned by {owner}")
            return False
        
        self.approvals[token_id] = approved
        
        # Добавляем событие
        event = TokenEvent(
            event_type="Approval",
            token_id=token_id,
            from_addr=owner,
            to_addr=approved,
            amount=None,
            metadata=token.metadata
        )
        token.add_event(event)
        self.events.append(event)
        
        logger.info(f"Approved {approved} for token {token_id}")
        return True
    
    def set_approval_for_all(self, operator: str, owner: str, approved: bool) -> bool:
        """Одобрение всех токенов оператору"""
        if approved:
            if owner not in self.operators:
                self.operators[owner] = []
            if operator not in self.operators[owner]:
                self.operators[owner].append(operator)
        else:
            if owner in self.operators and operator in self.operators[owner]:
                self.operators[owner].remove(operator)
        
        # Добавляем событие
        event = TokenEvent(
            event_type="ApprovalForAll",
            token_id=0,  # Для всех токенов
            from_addr=owner,
            to_addr=operator,
            amount=None,
            metadata=None
        )
        self.events.append(event)
        
        logger.info(f"Set approval for all tokens: {operator} -> {approved}")
        return True
    
    def get_approved(self, token_id: int) -> Optional[str]:
        """Получение одобренного адреса для токена"""
        return self.approvals.get(token_id)
    
    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        """Проверка одобрения всех токенов оператору"""
        return owner in self.operators and operator in self.operators[owner]
    
    def owner_of(self, token_id: int) -> Optional[str]:
        """Получение владельца токена"""
        token = self.tokens.get(token_id)
        return token.owner if token else None
    
    def balance_of(self, owner: str) -> int:
        """Получение баланса токенов владельца"""
        return len(self.owner_tokens.get(owner, []))
    
    def get_events(self, token_id: Optional[int] = None, 
                  event_type: Optional[str] = None) -> List[TokenEvent]:
        """Получение событий"""
        if token_id is not None:
            token = self.tokens.get(token_id)
            if not token:
                return []
            return token.get_events(event_type)
        
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events

class ERC1155:
    """Реализация стандарта ERC-1155 (Multi-Token)"""
    
    def __init__(self, name: str):
        self.name = name
        self.tokens: Dict[int, MultiToken] = {}
        self.operators: Dict[str, List[str]] = {}  # owner -> operator_addresses
        self.events: List[TokenEvent] = []
        self.collections: Dict[str, TokenCollection] = {}
        self.access_control = TokenAccessControl()
    
    def mint(self, to: str, token_id: int, token_type: int, 
             amount: Decimal, metadata: TokenMetadata, caller: str) -> bool:
        """Создание нового токена"""
        if not self.access_control.check_permission(caller, "can_mint"):
            logger.error(f"Caller {caller} does not have permission to mint")
            return False
        
        if not TokenValidator.validate_address(to):
            logger.error(f"Invalid address: {to}")
            return False
        
        if not TokenValidator.validate_token_id(token_id):
            logger.error(f"Invalid token ID: {token_id}")
            return False
        
        if not TokenValidator.validate_token_type(token_type):
            logger.error(f"Invalid token type: {token_type}")
            return False
        
        if not TokenValidator.validate_amount(amount):
            logger.error(f"Invalid amount: {amount}")
            return False
        
        if token_id in self.tokens:
            logger.error(f"Token {token_id} already exists")
            return False
        
        self.tokens[token_id] = MultiToken(
            token_id=token_id,
            token_type=token_type,
            supply=amount,
            metadata=metadata,
            owners={to: amount}
        )
        
        # Добавляем событие
        event = TokenEvent(
            event_type="Mint",
            token_id=token_id,
            from_addr=None,
            to_addr=to,
            amount=amount,
            metadata=metadata
        )
        self.tokens[token_id].add_event(event)
        self.events.append(event)
        
        logger.info(f"Minted token {token_id} to {to}")
        return True
    
    def create_collection(self, name: str, description: str, caller: str) -> str:
        """Создание коллекции токенов"""
        if not self.access_control.check_permission(caller, "can_create_collection"):
            logger.error(f"Caller {caller} does not have permission to create collections")
            return ""
        
        if name in self.collections:
            logger.error(f"Collection {name} already exists")
            return ""
        
        self.collections[name] = TokenCollection(
            name=name,
            description=description,
            tokens=[]
        )
        return name
    
    def add_to_collection(self, collection_name: str, token_id: int, caller: str) -> bool:
        """Добавление токена в коллекцию"""
        if not self.access_control.check_permission(caller, "can_manage_collection"):
            logger.error(f"Caller {caller} does not have permission to manage collections")
            return False
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return False
        
        return self.collections[collection_name].add_token(token_id)
    
    def remove_from_collection(self, collection_name: str, token_id: int) -> bool:
        """Удаление токена из коллекции"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return False
        
        return self.collections[collection_name].remove_token(token_id)
    
    def get_collection_tokens(self, collection_name: str) -> List[int]:
        """Получение токенов из коллекции"""
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} does not exist")
            return []
        
        return self.collections[collection_name].get_tokens()
    
    def get_collections(self) -> Dict[str, TokenCollection]:
        """Получение всех коллекций"""
        return self.collections.copy()
    
    def mint_batch(self, to: str, token_ids: List[int], token_types: List[int],
                   amounts: List[Decimal], metadatas: List[TokenMetadata], caller: str) -> bool:
        """Создание нескольких токенов"""
        if len(token_ids) != len(token_types) or len(token_ids) != len(amounts):
            logger.error("Invalid batch parameters")
            return False
        
        for i in range(len(token_ids)):
            if not self.mint(to, token_ids[i], token_types[i], amounts[i], metadatas[i], caller):
                return False
        
        return True
    
    def transfer(self, from_addr: str, to: str, token_id: int, 
                amount: Decimal, caller: str) -> bool:
        """Передача токенов"""
        if not self.access_control.check_permission(caller, "can_transfer"):
            logger.error(f"Caller {caller} does not have permission to transfer")
            return False
        
        if token_id not in self.tokens:
            logger.error(f"Token {token_id} does not exist")
            return False
        
        token = self.tokens[token_id]
        if from_addr not in token.owners or token.owners[from_addr] < amount:
            logger.error(f"Insufficient balance for token {token_id}")
            return False
        
        # Обновляем балансы
        token.owners[from_addr] -= amount
        if token.owners[from_addr] == 0:
            del token.owners[from_addr]
        
        if to not in token.owners:
            token.owners[to] = Decimal('0')
        token.owners[to] += amount
        
        # Добавляем событие
        event = TokenEvent(
            event_type="Transfer",
            token_id=token_id,
            from_addr=from_addr,
            to_addr=to,
            amount=amount,
            metadata=token.metadata
        )
        token.add_event(event)
        self.events.append(event)
        
        logger.info(f"Transferred {amount} of token {token_id} from {from_addr} to {to}")
        return True
    
    def transfer_batch(self, from_addr: str, to: str, token_ids: List[int],
                      amounts: List[Decimal], caller: str) -> bool:
        """Передача нескольких токенов"""
        if len(token_ids) != len(amounts):
            logger.error("Invalid batch parameters")
            return False
        
        for i in range(len(token_ids)):
            if not self.transfer(from_addr, to, token_ids[i], amounts[i], caller):
                return False
        
        return True
    
    def set_approval_for_all(self, operator: str, owner: str, approved: bool) -> bool:
        """Одобрение всех токенов оператору"""
        if approved:
            if owner not in self.operators:
                self.operators[owner] = []
            if operator not in self.operators[owner]:
                self.operators[owner].append(operator)
        else:
            if owner in self.operators and operator in self.operators[owner]:
                self.operators[owner].remove(operator)
        
        # Добавляем событие
        event = TokenEvent(
            event_type="ApprovalForAll",
            token_id=0,  # Для всех токенов
            from_addr=owner,
            to_addr=operator,
            amount=None,
            metadata=None
        )
        self.events.append(event)
        
        logger.info(f"Set approval for all tokens: {operator} -> {approved}")
        return True
    
    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        """Проверка одобрения всех токенов оператору"""
        return owner in self.operators and operator in self.operators[owner]
    
    def balance_of(self, owner: str, token_id: int) -> Decimal:
        """Получение баланса токена владельца"""
        token = self.tokens.get(token_id)
        if not token:
            return Decimal('0')
        return token.owners.get(owner, Decimal('0'))
    
    def balance_of_batch(self, owners: List[str], token_ids: List[int]) -> List[Decimal]:
        """Получение балансов нескольких токенов для нескольких владельцев"""
        if len(owners) != len(token_ids):
            logger.error("Invalid batch parameters")
            return []
        
        return [self.balance_of(owner, token_id) 
                for owner, token_id in zip(owners, token_ids)]
    
    def get_events(self, token_id: Optional[int] = None, 
                  event_type: Optional[str] = None) -> List[TokenEvent]:
        """Получение событий"""
        if token_id is not None:
            token = self.tokens.get(token_id)
            if not token:
                return []
            return token.get_events(event_type)
        
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events

# Пример использования
def example_usage():
    # Создание NFT контракта
    nft = ERC721("MyNFT", "MNFT")
    
    # Назначение ролей
    admin = "0x1234567890123456789012345678901234567890"
    minter = "0x0987654321098765432109876543210987654321"
    
    nft.access_control.grant_role(admin, TokenRole.ADMIN)
    nft.access_control.grant_role(minter, TokenRole.MINTER)
    
    # Создание NFT
    metadata = TokenMetadata(
        name="Cool NFT",
        description="A very cool NFT",
        image="https://example.com/image.png",
        attributes={"rarity": "legendary"}
    )
    
    nft.mint(admin, 1, metadata, minter)
    
    # Создание коллекции
    collection_name = nft.create_collection("Legendary NFTs", "Collection of legendary NFTs", admin)
    nft.add_to_collection(collection_name, 1, admin)
    
    # Получение токенов из коллекции
    tokens = nft.get_collection_tokens(collection_name)
    print(f"Tokens in collection: {tokens}")
    
    # Создание Multi-Token контракта
    multi_token = ERC1155("MyMultiToken")
    
    # Назначение ролей
    multi_token.access_control.grant_role(admin, TokenRole.ADMIN)
    multi_token.access_control.grant_role(minter, TokenRole.MINTER)
    
    # Создание токенов
    metadata1 = TokenMetadata(
        name="Gold Token",
        description="A gold token",
        image="https://example.com/gold.png",
        attributes={"type": "currency"}
    )
    
    metadata2 = TokenMetadata(
        name="Silver Token",
        description="A silver token",
        image="https://example.com/silver.png",
        attributes={"type": "currency"}
    )
    
    multi_token.mint_batch(
        admin,
        [1, 2],
        [1, 1],
        [Decimal('100'), Decimal('200')],
        [metadata1, metadata2],
        minter
    )
    
    # Создание коллекции
    collection_name = multi_token.create_collection("Currency Tokens", "Collection of currency tokens", admin)
    multi_token.add_to_collection(collection_name, 1, admin)
    multi_token.add_to_collection(collection_name, 2, admin)
    
    # Получение токенов из коллекции
    tokens = multi_token.get_collection_tokens(collection_name)
    print(f"Tokens in collection: {tokens}")

if __name__ == "__main__":
    example_usage() 