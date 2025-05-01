from typing import Dict, Set, Optional
from dataclasses import dataclass, field
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class Role:
    """Роль в системе доступа"""
    name: str
    admin_role: Optional[str] = None
    members: Set[str] = field(default_factory=set)

class AccessControl:
    """Базовый класс для управления доступом"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self._setup_default_roles()
    
    def _setup_default_roles(self):
        """Настройка ролей по умолчанию"""
        # Роль администратора
        self.roles["DEFAULT_ADMIN_ROLE"] = Role(
            name="DEFAULT_ADMIN_ROLE",
            admin_role=None
        )
        # Роль валидатора
        self.roles["VALIDATOR_ROLE"] = Role(
            name="VALIDATOR_ROLE",
            admin_role="DEFAULT_ADMIN_ROLE"
        )
        # Роль оператора
        self.roles["OPERATOR_ROLE"] = Role(
            name="OPERATOR_ROLE",
            admin_role="DEFAULT_ADMIN_ROLE"
        )
    
    def has_role(self, role: str, account: str) -> bool:
        """Проверка наличия роли у аккаунта"""
        if role not in self.roles:
            return False
        return account in self.roles[role].members
    
    def get_role_admin(self, role: str) -> Optional[str]:
        """Получение администратора роли"""
        if role not in self.roles:
            return None
        return self.roles[role].admin_role
    
    def grant_role(self, role: str, account: str, admin: str) -> bool:
        """Выдача роли аккаунту"""
        if not self.has_role(self.get_role_admin(role), admin):
            logger.error(f"Account {admin} is not authorized to grant role {role}")
            return False
        
        if role not in self.roles:
            logger.error(f"Role {role} does not exist")
            return False
        
        self.roles[role].members.add(account)
        logger.info(f"Role {role} granted to {account}")
        return True
    
    def revoke_role(self, role: str, account: str, admin: str) -> bool:
        """Отзыв роли у аккаунта"""
        if not self.has_role(self.get_role_admin(role), admin):
            logger.error(f"Account {admin} is not authorized to revoke role {role}")
            return False
        
        if role not in self.roles:
            logger.error(f"Role {role} does not exist")
            return False
        
        if account not in self.roles[role].members:
            logger.error(f"Account {account} does not have role {role}")
            return False
        
        self.roles[role].members.remove(account)
        logger.info(f"Role {role} revoked from {account}")
        return True
    
    def create_role(self, role: str, admin_role: str, creator: str) -> bool:
        """Создание новой роли"""
        if not self.has_role("DEFAULT_ADMIN_ROLE", creator):
            logger.error(f"Account {creator} is not authorized to create roles")
            return False
        
        if role in self.roles:
            logger.error(f"Role {role} already exists")
            return False
        
        if admin_role not in self.roles:
            logger.error(f"Admin role {admin_role} does not exist")
            return False
        
        self.roles[role] = Role(name=role, admin_role=admin_role)
        logger.info(f"Role {role} created with admin {admin_role}")
        return True

class AccessControlMixin:
    """Миксин для добавления управления доступом в контракты"""
    
    def __init__(self):
        self.access_control = AccessControl()
    
    def has_role(self, role: str, account: str) -> bool:
        """Проверка наличия роли у аккаунта"""
        return self.access_control.has_role(role, account)
    
    def get_role_admin(self, role: str) -> Optional[str]:
        """Получение администратора роли"""
        return self.access_control.get_role_admin(role)
    
    def grant_role(self, role: str, account: str, admin: str) -> bool:
        """Выдача роли аккаунту"""
        return self.access_control.grant_role(role, account, admin)
    
    def revoke_role(self, role: str, account: str, admin: str) -> bool:
        """Отзыв роли у аккаунта"""
        return self.access_control.revoke_role(role, account, admin)
    
    def create_role(self, role: str, admin_role: str, creator: str) -> bool:
        """Создание новой роли"""
        return self.access_control.create_role(role, admin_role, creator)

# Пример использования
def example_usage():
    # Создание контракта с управлением доступом
    contract = AccessControlMixin()
    
    # Инициализация ролей
    admin = "0x1234567890123456789012345678901234567890"
    operator = "0x1111111111111111111111111111111111111111"
    user = "0x2222222222222222222222222222222222222222"
    
    # Выдача ролей
    contract.grant_role("OPERATOR_ROLE", operator, admin)
    contract.grant_role("VALIDATOR_ROLE", operator, admin)
    
    # Проверка ролей
    print(f"Is operator: {contract.has_role('OPERATOR_ROLE', operator)}")
    print(f"Is validator: {contract.has_role('VALIDATOR_ROLE', operator)}")
    print(f"Is admin: {contract.has_role('DEFAULT_ADMIN_ROLE', operator)}")
    
    # Создание новой роли
    contract.create_role("CUSTOM_ROLE", "DEFAULT_ADMIN_ROLE", admin)
    contract.grant_role("CUSTOM_ROLE", user, admin)
    
    # Отзыв роли
    contract.revoke_role("CUSTOM_ROLE", user, admin)
    
    print(f"Has custom role: {contract.has_role('CUSTOM_ROLE', user)}")

@dataclass
class Ownable:
    """Миксин для управления владельцем"""
    owner: str
    pending_owner: Optional[str] = None
    
    def is_owner(self, account: str) -> bool:
        """Проверка владельца"""
        return self.owner == account
    
    def transfer_ownership(self, new_owner: str, current_owner: str) -> bool:
        """Передача владения"""
        if not self.is_owner(current_owner):
            logger.error(f"Caller {current_owner} is not the owner")
            return False
        
        self.pending_owner = new_owner
        logger.info(f"Ownership transfer initiated to {new_owner}")
        return True
    
    def accept_ownership(self, new_owner: str) -> bool:
        """Принятие владения"""
        if self.pending_owner != new_owner:
            logger.error(f"Caller {new_owner} is not the pending owner")
            return False
        
        self.owner = new_owner
        self.pending_owner = None
        logger.info(f"Ownership transferred to {new_owner}")
        return True

if __name__ == "__main__":
    example_usage() 