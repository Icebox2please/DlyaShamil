from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import re
import json
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    """Ошибка валидации"""
    field: str
    message: str
    value: Any

class TokenValidator:
    """Валидатор токенов"""
    
    # Регулярные выражения для валидации
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// или https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # домен
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # порт
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    ADDRESS_PATTERN = re.compile(r'^0x[a-fA-F0-9]{40}$')
    
    # Ограничения
    MAX_NAME_LENGTH = 100
    MAX_DESCRIPTION_LENGTH = 1000
    MAX_ATTRIBUTES = 50
    MAX_ATTRIBUTE_KEY_LENGTH = 50
    MAX_ATTRIBUTE_VALUE_LENGTH = 200
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> List[ValidationError]:
        """Валидация метаданных токена"""
        errors = []
        
        # Проверка обязательных полей
        required_fields = ['name', 'description', 'image']
        for field in required_fields:
            if field not in metadata:
                errors.append(ValidationError(
                    field=field,
                    message=f"Missing required field: {field}",
                    value=None
                ))
        
        # Валидация имени
        if 'name' in metadata:
            name = metadata['name']
            if not isinstance(name, str):
                errors.append(ValidationError(
                    field='name',
                    message="Name must be a string",
                    value=name
                ))
            elif len(name) > cls.MAX_NAME_LENGTH:
                errors.append(ValidationError(
                    field='name',
                    message=f"Name length exceeds maximum of {cls.MAX_NAME_LENGTH} characters",
                    value=name
                ))
        
        # Валидация описания
        if 'description' in metadata:
            description = metadata['description']
            if not isinstance(description, str):
                errors.append(ValidationError(
                    field='description',
                    message="Description must be a string",
                    value=description
                ))
            elif len(description) > cls.MAX_DESCRIPTION_LENGTH:
                errors.append(ValidationError(
                    field='description',
                    message=f"Description length exceeds maximum of {cls.MAX_DESCRIPTION_LENGTH} characters",
                    value=description
                ))
        
        # Валидация изображения
        if 'image' in metadata:
            image = metadata['image']
            if not isinstance(image, str):
                errors.append(ValidationError(
                    field='image',
                    message="Image must be a string",
                    value=image
                ))
            elif not cls.URL_PATTERN.match(image):
                errors.append(ValidationError(
                    field='image',
                    message="Image must be a valid URL",
                    value=image
                ))
        
        # Валидация атрибутов
        if 'attributes' in metadata:
            attributes = metadata['attributes']
            if not isinstance(attributes, dict):
                errors.append(ValidationError(
                    field='attributes',
                    message="Attributes must be a dictionary",
                    value=attributes
                ))
            elif len(attributes) > cls.MAX_ATTRIBUTES:
                errors.append(ValidationError(
                    field='attributes',
                    message=f"Number of attributes exceeds maximum of {cls.MAX_ATTRIBUTES}",
                    value=attributes
                ))
            else:
                for key, value in attributes.items():
                    if not isinstance(key, str):
                        errors.append(ValidationError(
                            field=f'attributes.{key}',
                            message="Attribute key must be a string",
                            value=key
                        ))
                    elif len(key) > cls.MAX_ATTRIBUTE_KEY_LENGTH:
                        errors.append(ValidationError(
                            field=f'attributes.{key}',
                            message=f"Attribute key length exceeds maximum of {cls.MAX_ATTRIBUTE_KEY_LENGTH} characters",
                            value=key
                        ))
                    
                    if isinstance(value, str) and len(value) > cls.MAX_ATTRIBUTE_VALUE_LENGTH:
                        errors.append(ValidationError(
                            field=f'attributes.{key}',
                            message=f"Attribute value length exceeds maximum of {cls.MAX_ATTRIBUTE_VALUE_LENGTH} characters",
                            value=value
                        ))
        
        return errors
    
    @classmethod
    def validate_address(cls, address: str) -> bool:
        """Валидация адреса"""
        return bool(cls.ADDRESS_PATTERN.match(address))
    
    @classmethod
    def validate_amount(cls, amount: Union[int, Decimal]) -> bool:
        """Валидация суммы"""
        try:
            if isinstance(amount, str):
                amount = Decimal(amount)
            return amount > 0
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def validate_token_id(cls, token_id: int) -> bool:
        """Валидация ID токена"""
        return isinstance(token_id, int) and token_id > 0
    
    @classmethod
    def validate_token_type(cls, token_type: int) -> bool:
        """Валидация типа токена"""
        return isinstance(token_type, int) and token_type >= 0
    
    @classmethod
    def validate_metadata_json(cls, json_str: str) -> List[ValidationError]:
        """Валидация JSON метаданных"""
        try:
            metadata = json.loads(json_str)
            return cls.validate_metadata(metadata)
        except json.JSONDecodeError as e:
            return [ValidationError(
                field='metadata',
                message=f"Invalid JSON: {str(e)}",
                value=json_str
            )]

# Пример использования
def example_usage():
    # Валидация метаданных
    metadata = {
        "name": "Cool NFT",
        "description": "A very cool NFT",
        "image": "https://example.com/image.png",
        "attributes": {
            "rarity": "legendary",
            "power": 100
        }
    }
    
    errors = TokenValidator.validate_metadata(metadata)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error.field}: {error.message}")
    else:
        print("Metadata is valid")
    
    # Валидация адреса
    address = "0x1234567890123456789012345678901234567890"
    if TokenValidator.validate_address(address):
        print("Address is valid")
    else:
        print("Address is invalid")
    
    # Валидация суммы
    amount = Decimal("100.5")
    if TokenValidator.validate_amount(amount):
        print("Amount is valid")
    else:
        print("Amount is invalid")

if __name__ == "__main__":
    example_usage() 