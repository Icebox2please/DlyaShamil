import json
import time
import random
import logging
from crypto import Crypto
import hashlib
from typing import Dict, List, Optional, Any, Union
import requests

logger = logging.getLogger(__name__)

class BlockEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Block):
            return obj.to_dict()
        return super().default(obj)

class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, validator, signature, epoch):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.validator = validator
        self.signature = signature
        self.epoch = epoch
        self.crypto = Crypto()
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "validator": self.validator,
            "signature": self.signature,
            "epoch": self.epoch
        }, sort_keys=True)
        return self.crypto.sha256(block_string)

    def to_dict(self):
        return {
            "index": self.index,
            "transactions": self.transactions,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "validator": self.validator,
            "signature": self.signature,
            "epoch": self.epoch
        }

    def copy(self):
        """Creates a copy of the block"""
        return Block(
            index=self.index,
            transactions=self.transactions.copy(),
            timestamp=self.timestamp,
            previous_hash=self.previous_hash,
            validator=self.validator,
            signature=self.signature,
            epoch=self.epoch
        )

    def __getitem__(self, key):
        """Makes the block subscriptable"""
        if key == 'index':
            return self.index
        elif key == 'transactions':
            return self.transactions
        elif key == 'timestamp':
            return self.timestamp
        elif key == 'previous_hash':
            return self.previous_hash
        elif key == 'validator':
            return self.validator
        elif key == 'signature':
            return self.signature
        elif key == 'epoch':
            return self.epoch
        elif key == 'hash':
            return self.hash
        else:
            raise KeyError(f"Block has no attribute '{key}'")

    def __setitem__(self, key, value):
        """Allows item assignment"""
        if key == 'index':
            self.index = value
        elif key == 'transactions':
            self.transactions = value
        elif key == 'timestamp':
            self.timestamp = value
        elif key == 'previous_hash':
            self.previous_hash = value
        elif key == 'validator':
            self.validator = value
        elif key == 'signature':
            self.signature = value
        elif key == 'epoch':
            self.epoch = value
        elif key == 'hash':
            self.hash = value
        else:
            raise KeyError(f"Block has no attribute '{key}'")

    def __str__(self):
        """String representation of the block"""
        return json.dumps(self.to_dict(), cls=BlockEncoder)

    def __repr__(self):
        """Representation of the block"""
        return f"Block(index={self.index}, hash={self.hash})"

    def __json__(self):
        """JSON serialization support"""
        return self.to_dict()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.validators = {}
        self.contract_states = {}
        self.contract_events = {}
        self.state = {}
        self.gas_pool = {}
        self.epoch_length = 100
        self.current_epoch = 0
        self.slash_conditions = {
            'double_sign': 0.1,  # 10% штраф за двойную подпись
            'invalid_block': 0.05,  # 5% штраф за невалидный блок
            'late_block': 0.02,  # 2% штраф за опоздание
            'network_attack': 0.2  # 20% штраф за сетевые атаки
        }
        self.validators_history = {}
        self.fork_resolution_depth = 6
        self.block_time = 10
        self.max_validators = 21  # Максимальное количество валидаторов
        self.min_stake_ratio = 0.01  # Минимальная доля стейка для валидатора
        self.network_attack_threshold = 0.33  # Порог для определения сетевой атаки
        self.validator_blacklist = set()  # Черный список валидаторов
        self.validator_penalties = {}
        self.validator_rewards = {}
        self.crypto = Crypto()
        self.block_processing_delay = 5  # Увеличиваем задержку до 5 секунд
        self.create_genesis_block()
        logging.info("Blockchain initialization completed")

    def create_genesis_block(self):
        # Фиксированное время для генезис-блока
        genesis_timestamp = 0  # Начало эпохи Unix
        
        genesis_block = Block(
            index=0,
            timestamp=genesis_timestamp,
            transactions=[],
            previous_hash='0' * 64,
            validator='genesis',
            signature='0',
            epoch=0
        )
        self.chain.append(genesis_block)
        self.contract_states = {}
        self.contract_events = {}
        self.gas_pool = {}
        logging.info("Genesis block created")

    def start_new_epoch(self):
        """Начало новой эпохи"""
        self.current_epoch += 1
        logging.info(f"Starting new epoch {self.current_epoch}")
        
        # Выбор валидаторов на новую эпоху
        new_validators = self.select_validators()
        
        # Обновление стейков
        self.update_stakes()
        
        # Очистка истории
        self.clear_validator_history()
        
        return new_validators

    def select_validators(self) -> Dict[str, float]:
        """Улучшенный выбор валидаторов с защитой от атак"""
        try:
            # Фильтрация черного списка
            eligible_validators = {
                v: s for v, s in self.validators.items()
                if v not in self.validator_blacklist
            }
            
            total_stake = sum(eligible_validators.values())
            if total_stake == 0:
                return {}
            
            # Использование хеша последнего блока для детерминированности
            seed = int(self.chain[-1]['hash'], 16)
            random.seed(seed)
            
            # Выбор валидаторов с учетом защиты от атак
            selected = {}
            remaining_stake = total_stake
            
            while len(selected) < min(len(eligible_validators), self.max_validators):
                target = random.uniform(0, remaining_stake)
                current = 0
                
                for validator, stake in eligible_validators.items():
                    if validator in selected:
                        continue
                    
                    # Проверка минимальной доли стейка
                    if stake / total_stake < self.min_stake_ratio:
                        continue
                    
                    current += stake
                    if current >= target:
                        selected[validator] = stake
                        remaining_stake -= stake
                        break
            
            return selected
        except Exception as e:
            logging.error(f"Validator selection error: {str(e)}")
            return {}

    def update_stakes(self):
        """Обновление стейков валидаторов"""
        for validator in self.validators:
            # Начисление вознаграждения
            reward = self.calculate_reward(validator)
            self.validators[validator] += reward
            
            # Применение штрафов
            penalties = self.calculate_penalties(validator)
            self.validators[validator] -= penalties
            
            logging.info(f"Validator {validator} updated: reward={reward}, penalties={penalties}")

    def calculate_reward(self, validator: str) -> float:
        """Расчет вознаграждения валидатора"""
        base_reward = 1.0
        participation_rate = self.get_validator_participation(validator)
        return base_reward * participation_rate

    def calculate_penalties(self, validator: str) -> float:
        """Расчет штрафов валидатора"""
        total_penalty = 0
        for condition, rate in self.slash_conditions.items():
            violations = self.count_validator_violations(validator, condition)
            total_penalty += self.validators[validator] * rate * violations
        return total_penalty

    def slash_validator(self, address: str, reason: str):
        """Наказание валидатора"""
        if address not in self.validators:
            return
            
        stake = self.validators[address]
        penalty = stake * self.slash_conditions.get(reason, 0)
        
        # Уменьшение стейка
        self.validators[address] -= penalty
        
        # Запись в историю
        if address not in self.validators_history:
            self.validators_history[address] = []
            
        self.validators_history[address].append({
            'type': 'slash',
            'reason': reason,
            'amount': penalty,
            'timestamp': time.time()
        })
        
        logging.warning(f"Validator {address} slashed: {reason}, penalty={penalty}")

    def clear_validator_history(self):
        """Очистка истории валидаторов"""
        self.validators_history = {}

    def get_validator_participation(self, validator: str) -> float:
        """Расчет участия валидатора"""
        if validator not in self.validators_history:
            return 1.0
            
        history = self.validators_history[validator]
        total_blocks = len(self.chain)
        if total_blocks == 0:
            return 1.0
            
        missed_blocks = sum(1 for event in history if event['type'] == 'missed_block')
        return 1.0 - (missed_blocks / total_blocks)

    def count_validator_violations(self, validator: str, violation_type: str) -> int:
        """Подсчет нарушений валидатора"""
        if validator not in self.validators_history:
            return 0
            
        return sum(1 for event in self.validators_history[validator] 
                  if event['type'] == 'slash' and event['reason'] == violation_type)

    def validate_block(self, block: Union[Dict[str, Any], Block]) -> bool:
        """Проверка валидности блока"""
        try:
            # Если блок - это объект Block, преобразуем его в словарь
            if isinstance(block, Block):
                block = block.__json__()
            
            # Проверяем наличие всех необходимых полей
            required_fields = ['index', 'timestamp', 'transactions', 'previous_hash', 'validator', 'signature', 'epoch']
            if not all(field in block for field in required_fields):
                logger.error("Missing required fields in block")
                return False

            # Проверяем подпись валидатора
            if not self.verify_validator_signature(block):
                logger.error("Invalid validator signature")
                return False

            # Проверяем временную метку
            if not self.validate_timestamp(block['timestamp']):
                logger.error("Invalid timestamp")
                return False

            # Проверяем транзакции
            for tx in block['transactions']:
                if not self.validate_transaction(tx):
                    logger.error(f"Invalid transaction: {tx}")
                    return False

            # Проверяем эпоху
            if block['epoch'] != self.current_epoch:
                logger.error(f"Invalid epoch: {block['epoch']} != {self.current_epoch}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating block: {str(e)}")
            return False

    def verify_validator_signature(self, block: Union[Dict[str, Any], Block]) -> bool:
        """Проверка подписи валидатора"""
        try:
            # Проверяем, что блок является объектом Block
            if isinstance(block, Block):
                block = block.to_dict()
            
            # Проверяем наличие всех необходимых полей
            required_fields = ['index', 'timestamp', 'transactions', 'previous_hash', 'validator', 'signature']
            if not all(field in block for field in required_fields):
                logger.error("Missing required fields in block")
                return False
            
            # Для генезис-блока пропускаем проверку подписи
            if block['index'] == 0:
                return True
            
            # Проверяем, что валидатор существует
            if block['validator'] not in self.validators:
                logger.error(f"Validator {block['validator']} not found")
                return False
            
            # Создаем строку для проверки подписи
            block_data = {
                'index': block['index'],
                'timestamp': block['timestamp'],
                'transactions': block['transactions'],
                'previous_hash': block['previous_hash'],
                'validator': block['validator']
            }
            data_string = json.dumps(block_data, sort_keys=True)
            
            # Проверяем подпись
            return self.crypto.verify(data_string, block['signature'], block['validator'])
            
        except Exception as e:
            logger.error(f"Error verifying validator signature: {str(e)}")
            return False

    def validate_timestamp(self, timestamp: float) -> bool:
        """Валидация временной метки блока"""
        try:
            current_time = time.time()
            last_block_time = self.chain[-1]['timestamp'] if self.chain else 0
            time_diff = abs(current_time - timestamp)
            max_allowed_diff = 300  # 5 минут
            
            logger.info(f"Validating timestamp: current={current_time}, last={last_block_time}, diff={time_diff}, max_allowed={max_allowed_diff}")
            
            if time_diff > max_allowed_diff:
                logger.error(f"Invalid timestamp: time difference {time_diff} is not in range [0, {max_allowed_diff}]")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating timestamp: {str(e)}")
            return False

    def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Валидация транзакции"""
        try:
            # Проверяем наличие базовых полей
            if 'type' not in transaction or 'timestamp' not in transaction:
                logger.error("Missing required fields in transaction")
                return False

            # Валидация в зависимости от типа транзакции
            if transaction['type'] == 'transfer':
                required_fields = ['from', 'to', 'amount']
                if not all(field in transaction for field in required_fields):
                    logger.error("Missing required fields in transfer transaction")
                    return False
                    
                if not transaction['from'] or not transaction['to']:
                    logger.error("Sender and recipient addresses are required")
                    return False
                    
                if not isinstance(transaction['amount'], (int, float)) or transaction['amount'] <= 0:
                    logger.error("Invalid amount")
                    return False
                    
                if transaction['from'] not in self.state:
                    logger.error("Sender does not exist")
                    return False
                    
                if self.state[transaction['from']] < transaction['amount']:
                    logger.error("Insufficient balance")
                    return False
                    
            elif transaction['type'] == 'validator':
                required_fields = ['address', 'stake']
                if not all(field in transaction for field in required_fields):
                    logger.error("Missing required fields in validator transaction")
                    return False
                    
                if not transaction['address']:
                    logger.error("Validator address is required")
                    return False
                    
                if not isinstance(transaction['stake'], (int, float)) or transaction['stake'] <= 0:
                    logger.error("Invalid stake amount")
                    return False
                    
            elif transaction['type'] == 'contract_deploy':
                required_fields = ['name', 'code']
                if not all(field in transaction for field in required_fields):
                    logger.error("Missing required fields in contract deployment transaction")
                    return False
                    
                if not transaction['name'] or not transaction['code']:
                    logger.error("Contract name and code are required")
                    return False
                    
                if transaction['name'] in self.contract_states:
                    logger.error("Contract with this name already exists")
                    return False
                    
            elif transaction['type'] == 'contract_execution':
                required_fields = ['contract_name', 'function', 'args']
                if not all(field in transaction for field in required_fields):
                    logger.error("Missing required fields in contract execution transaction")
                    return False
                    
                if not transaction['contract_name'] or not transaction['function']:
                    logger.error("Contract name and function are required")
                    return False
                    
                if transaction['contract_name'] not in self.contract_states:
                    logger.error("Contract does not exist")
                    return False
                    
                contract = self.contract_states[transaction['contract_name']]
                if transaction['function'] not in contract:
                    logger.error("Function does not exist in contract")
                    return False
                    
            else:
                logger.error(f"Invalid transaction type: {transaction['type']}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating transaction: {str(e)}")
            return False

    def validate_transfer(self, transaction: Dict[str, Any]) -> bool:
        """Валидация транзакции перевода"""
        try:
            required_fields = ['type', 'from', 'to', 'amount', 'timestamp']
            if not all(field in transaction for field in required_fields):
                logger.error("Missing required fields in transaction")
                return False
                
            if transaction['type'] != 'transfer':
                logger.error("Invalid transaction type")
                return False
                
            if transaction['from'] not in self.state:
                self.state[transaction['from']] = 0
            if transaction['to'] not in self.state:
                self.state[transaction['to']] = 0
                
            if self.state[transaction['from']] < transaction['amount']:
                logger.error(f"Insufficient balance for {transaction['from']}")
                return False
                
            if transaction['amount'] <= 0:
                logger.error("Invalid amount")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating transaction: {str(e)}")
            return False

    def validate_contract_deploy(self, transaction: Dict[str, Any]) -> bool:
        """Валидация транзакции развертывания контракта"""
        try:
            required_fields = ['type', 'name', 'code', 'timestamp']
            if not all(field in transaction for field in required_fields):
                logger.error("Missing required fields in contract deployment transaction")
                return False
                
            if transaction['type'] != 'contract_deploy':
                logger.error("Invalid transaction type for contract deployment")
                return False
                
            if not transaction['name'] or not transaction['code']:
                logger.error("Contract name and code are required")
                return False
                
            if transaction['name'] in self.contract_states:
                logger.error("Contract with this name already exists")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating contract deployment: {str(e)}")
            return False

    def validate_contract_execution(self, transaction: Dict[str, Any]) -> bool:
        """Валидация транзакции выполнения контракта"""
        try:
            required_fields = ['type', 'contract_name', 'function', 'args', 'timestamp']
            if not all(field in transaction for field in required_fields):
                logger.error("Missing required fields in contract execution transaction")
                return False
                
            if transaction['type'] != 'contract_execution':
                logger.error("Invalid transaction type for contract execution")
                return False
                
            if not transaction['contract_name'] or not transaction['function']:
                logger.error("Contract name and function are required")
                return False
                
            if transaction['contract_name'] not in self.contract_states:
                logger.error("Contract does not exist")
                return False
                
            contract = self.contract_states[transaction['contract_name']]
            if transaction['function'] not in contract:
                logger.error("Function does not exist in contract")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating contract execution: {str(e)}")
            return False

    def validate_validator(self, transaction: Dict[str, Any]) -> bool:
        """Валидация транзакции добавления валидатора"""
        try:
            required_fields = ['type', 'address', 'stake', 'timestamp']
            if not all(field in transaction for field in required_fields):
                logger.error("Missing required fields in validator transaction")
                return False
                
            if transaction['type'] != 'validator':
                logger.error("Invalid transaction type")
                return False
                
            if not transaction['address']:
                logger.error("Validator address is required")
                return False
                
            if not isinstance(transaction['stake'], (int, float)) or transaction['stake'] <= 0:
                logger.error("Invalid stake amount")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating validator transaction: {str(e)}")
            return False

    def resolve_fork(self, chain1: List[Dict], chain2: List[Dict]) -> List[Dict]:
        """Разрешение форка между двумя цепочками"""
        # Находим общий предок
        common_ancestor = self.find_common_ancestor(chain1, chain2)
        if not common_ancestor:
            return []
            
        # Выбираем цепочку с большей суммарной сложностью
        chain1_difficulty = self.calculate_chain_difficulty(chain1, common_ancestor)
        chain2_difficulty = self.calculate_chain_difficulty(chain2, common_ancestor)
        
        return chain1 if chain1_difficulty > chain2_difficulty else chain2

    def find_common_ancestor(self, chain1: List[Dict], chain2: List[Dict]) -> Optional[Dict]:
        """Поиск общего предка двух цепочек"""
        for block1 in reversed(chain1):
            for block2 in reversed(chain2):
                if block1['hash'] == block2['hash']:
                    return block1
        return None

    def calculate_chain_difficulty(self, chain: List[Dict], ancestor: Dict) -> float:
        """Расчет сложности цепочки"""
        difficulty = 0
        for block in chain:
            if block['hash'] == ancestor['hash']:
                break
            difficulty += self.calculate_block_difficulty(block)
        return difficulty

    def calculate_block_difficulty(self, block: Dict) -> float:
        """Расчет сложности блока"""
        # Сложность зависит от стейка валидатора и времени создания блока
        validator_stake = self.validators.get(block['validator'], 0)
        time_diff = block['timestamp'] - self.chain[-1]['timestamp']
        
        return validator_stake * (1 / max(time_diff, 1))

    def sync_validators(self, peer: Dict) -> None:
        """Синхронизация валидаторов с пиром"""
        try:
            # Получаем актуальный список валидаторов от пира
            response = requests.get(f"http://{peer['address']}:{peer['port']}/api/validators")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    peer_validators = data.get('validators', {})
                    
                    # Обновляем локальный список валидаторов
                    for address, stake in peer_validators.items():
                        if stake > 0:  # Проверяем, что ставка валидатора положительная
                            if address not in self.validators or self.validators[address] != stake:
                                self.validators[address] = stake
                                logger.info(f"Updated validator {address} with stake {stake}")
                else:
                    logger.error(f"Failed to get validators from peer {peer['address']}:{peer['port']}: {data.get('error')}")
            else:
                logger.error(f"Failed to get validators from peer {peer['address']}:{peer['port']}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error syncing validators: {str(e)}")

    def sync_with_peer(self, peer: Dict) -> bool:
        """Синхронизация с пиром"""
        try:
            # Запрос блоков
            blocks = self.request_blocks(peer)
            if not blocks:
                return False
                
            # Проверка цепочки
            if not self.validate_chain(blocks):
                return False
                
            # Применение блоков
            self.apply_blocks(blocks)
            
            # Синхронизация валидаторов
            self.sync_validators(peer)
            
            # Синхронизация состояния контрактов
            self.sync_contract_states(peer)
            
            return True
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            return False

    def request_blocks(self, peer: Dict) -> List[Dict]:
        """Запрос блоков у пира"""
        # Здесь должна быть реализация запроса блоков
        return []

    def validate_chain(self, blocks: List[Dict]) -> bool:
        """Проверка цепочки блоков"""
        for i in range(1, len(blocks)):
            if not self.validate_block(blocks[i]):
                return False
            if blocks[i]['previous_hash'] != blocks[i-1]['hash']:
                return False
        return True

    def apply_blocks(self, blocks: List[Dict]):
        """Применение блоков"""
        for block in blocks:
            self.chain.append(block)
            self.apply_transactions(block['transactions'])

    def apply_transactions(self, transactions: List[Dict]):
        """Применяет транзакции из блока"""
        for tx in transactions:
            if tx['type'] == 'transfer':
                self.apply_transfer(tx)
            elif tx['type'] == 'contract_deploy':
                self.apply_contract_deploy(tx)
            elif tx['type'] == 'contract_execution':
                self.apply_contract_execution(tx)
            elif tx['type'] == 'validator':
                self.apply_validator(tx['address'], tx['stake'])
            else:
                logger.warning(f"Unknown transaction type: {tx['type']}")

    def apply_transfer(self, tx: Dict):
        """Применение транзакции перевода"""
        self.state[tx['from']] -= tx['amount']
        self.state[tx['to']] = self.state.get(tx['to'], 0) + tx['amount']

    def apply_contract_deploy(self, tx: Dict):
        """Применение деплоя контракта"""
        try:
            contract_name = tx['name']
            code = tx['code']
            
            # Инициализация состояния контракта
            self.contract_states[contract_name] = {
                'code': code,
                'counter': 0,  # Начальное значение счетчика
                'last_updated': tx['timestamp']
            }
            
            # Инициализация событий контракта
            self.contract_events[contract_name] = []
            
            logger.info(f"Contract {contract_name} deployed successfully")
            
        except Exception as e:
            logger.error(f"Error applying contract deployment: {str(e)}")
            raise

    def apply_contract_execution(self, tx: Dict):
        """Применение выполнения контракта"""
        try:
            contract_name = tx['contract']
            function_name = tx['function']
            args = tx['args']
            
            if contract_name not in self.contract_states:
                raise ValueError(f"Contract {contract_name} not found")
                
            contract_state = self.contract_states[contract_name]
            
            # Выполняем функцию контракта
            if function_name == 'increment':
                contract_state['counter'] = contract_state.get('counter', 0) + 1
                result = contract_state['counter']
            elif function_name == 'decrement':
                contract_state['counter'] = contract_state.get('counter', 0) - 1
                result = contract_state['counter']
            elif function_name == 'get_counter':
                result = contract_state.get('counter', 0)
            else:
                raise ValueError(f"Unknown function {function_name}")
                
            # Обновляем состояние контракта
            self.contract_states[contract_name] = contract_state
            
            # Добавляем событие
            if contract_name not in self.contract_events:
                self.contract_events[contract_name] = []
            self.contract_events[contract_name].append({
                'function': function_name,
                'args': args,
                'result': result,
                'timestamp': tx['timestamp']
            })
            
            logger.info(f"Contract {contract_name} function {function_name} executed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error applying contract execution: {str(e)}")
            raise

    def apply_validator(self, address: str, stake: float) -> None:
        """Применение валидатора после подтверждения транзакции"""
        try:
            logger.info(f"Attempting to apply validator: {address} with stake {stake}")
            logger.info(f"Current validators before applying: {self.validators}")
            
            self.validators[address] = stake
            logger.info(f"Successfully applied validator: {address} with stake {stake}")
            logger.info(f"Current validators after applying: {self.validators}")
        except Exception as e:
            logger.error(f"Error applying validator: {str(e)}")

    def sync_contract_states(self, peer: Dict):
        """Синхронизация состояния контрактов"""
        # Здесь должна быть реализация синхронизации состояния
        pass

    def upgrade_contract(self, name: str, new_code: str) -> bool:
        """Обновление контракта"""
        if name not in self.contract_states:
            return False
            
        # Проверка прав на обновление
        if not self.has_upgrade_permission(name):
            return False
            
        # Создание новой версии
        new_version = {
            'code': new_code,
            'timestamp': time.time(),
            'upgraded_from': self.contract_states[name].get('version', 0)
        }
        
        # Миграция состояния
        self.migrate_state(name, new_version)
        
        return True

    def has_upgrade_permission(self, name: str) -> bool:
        """Проверка прав на обновление контракта"""
        # Здесь должна быть реализация проверки прав
        return True

    def migrate_state(self, name: str, new_version: Dict):
        """Миграция состояния контракта"""
        old_state = self.contract_states[name]['state']
        # Здесь должна быть реализация миграции состояния
        self.contract_states[name] = {
            'code': new_version['code'],
            'state': old_state,
            'version': new_version['upgraded_from'] + 1,
            'last_updated': new_version['timestamp']
        }

    def add_transaction(self, sender: str, recipient: str, amount: float) -> bool:
        """Добавление новой транзакции в пул"""
        try:
            # Проверка баланса отправителя
            if sender not in self.state:
                self.state[sender] = 0
            if recipient not in self.state:
                self.state[recipient] = 0
                
            if self.state[sender] < amount:
                logger.error(f"Insufficient balance for {sender}")
                return False
                
            transaction = {
                'type': 'transfer',
                'from': sender,
                'to': recipient,
                'amount': amount,
                'timestamp': time.time()
            }
            
            self.current_transactions.append(transaction)
            logger.info(f"Transaction added: {sender} -> {recipient} ({amount})")
            return True
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return False

    def add_contract(self, name, code, gas_limit=None):
        """Добавляет смарт-контракт в пул с указанием лимита газа"""
        if name in self.contract_states:
            raise ValueError(f"Contract {name} already exists")
            
        contract = {
            'type': 'contract_deploy',
            'name': name,
            'code': code,
            'gas_limit': gas_limit,
            'timestamp': time.time()
        }
        self.current_transactions.append(contract)
        logging.info(f"Contract deployment added to pool: {name} with gas limit {gas_limit}")
        return len(self.chain)

    def execute_contract(self, name: str, function: str, args: List[Any], gas_limit: Optional[int] = None) -> Any:
        """Выполнение смарт-контракта"""
        try:
            if name not in self.contract_states:
                raise ValueError(f"Contract {name} not found")
                
            contract_state = self.contract_states[name]
            
            # Создаем транзакцию для выполнения контракта
            tx = {
                'type': 'contract_execution',
                'contract': name,
                'function': function,
                'args': args,
                'timestamp': time.time()
            }
            
            # Добавляем транзакцию в пул
            self.current_transactions.append(tx)
            
            # Ждем, пока транзакция будет обработана
            while tx in self.current_transactions:
                time.sleep(0.1)
                
            # Получаем результат из состояния контракта
            if name in self.contract_states:
                contract_state = self.contract_states[name]
                if function in contract_state:
                    return contract_state[function]
                else:
                    raise ValueError(f"Function {function} not found in contract {name}")
            else:
                raise ValueError(f"Contract {name} not found after execution")
                
        except Exception as e:
            logger.error(f"Error executing contract: {str(e)}")
            raise

    def _reserve_gas(self, transaction, gas_amount):
        """Резервирует газ для транзакции"""
        # Здесь должна быть логика проверки баланса и резервирования газа
        # В реальной системе это будет связано с балансами пользователей
        self.gas_pool[id(transaction)] = gas_amount

    def _refund_gas(self, transaction, used_gas):
        """Возвращает неиспользованный газ"""
        if id(transaction) in self.gas_pool:
            reserved = self.gas_pool[id(transaction)]
            refund = reserved - used_gas
            if refund > 0:
                # Здесь должна быть логика возврата газа пользователю
                del self.gas_pool[id(transaction)]
                return refund
        return 0

    def add_validator(self, address: str, stake: float) -> bool:
        """Добавление валидатора"""
        try:
            logger.info(f"Attempting to add validator: {address} with stake {stake}")
            logger.info(f"Current validators before adding: {self.validators}")
            
            # Создаем транзакцию для добавления валидатора
            transaction = {
                'type': 'validator',
                'address': address,
                'stake': stake,
                'timestamp': time.time()
            }
            
            self.current_transactions.append(transaction)
            logger.info(f"Added validator transaction: {address} with stake {stake}")
            logger.info(f"Current transactions after adding: {self.current_transactions}")
            return True
        except Exception as e:
            logger.error(f"Error adding validator: {str(e)}")
            return False

    def mine_block(self, validator_address: str, validator_private_key: str) -> Optional[Dict[str, Any]]:
        """Майнинг нового блока"""
        try:
            logger.info(f"Starting to mine block with validator: {validator_address}")
            logger.info(f"Current validators during mining: {self.validators}")
            
            # Проверяем, есть ли транзакция добавления валидатора
            validator_tx = next(
                (tx for tx in self.current_transactions 
                 if tx['type'] == 'validator' and tx['address'] == validator_address),
                None
            )
            
            # Если валидатора нет в списке и нет транзакции добавления
            if validator_address not in self.validators and not validator_tx:
                logger.error(f"Validator {validator_address} not found and no pending transaction")
                return None
            
            # Если есть транзакция добавления валидатора, применяем её
            if validator_tx:
                logger.info(f"Applying validator transaction for {validator_address}")
                self.apply_validator(validator_tx['address'], validator_tx['stake'])
            
            # Создаем новый блок
            last_block = self.chain[-1]
            new_block = Block(
                index=last_block['index'] + 1,
                transactions=self.current_transactions.copy(),
                timestamp=time.time(),
                previous_hash=self.hash_block(last_block),
                validator=validator_address,
                signature='',
                epoch=self.current_epoch
            )
            
            # Подписываем блок
            block_data = {
                'index': new_block['index'],
                'timestamp': new_block['timestamp'],
                'transactions': new_block['transactions'],
                'previous_hash': new_block['previous_hash'],
                'validator': new_block['validator']
            }
            block_str = json.dumps(block_data, sort_keys=True)
            new_block['signature'] = self.crypto.sign(block_str, validator_private_key)
            
            # Проверяем валидность блока
            if not self.validate_block(new_block):
                logger.error("Invalid block created")
                return None
            
            # Добавляем блок в цепочку
            self.chain.append(new_block)
            self.current_transactions = []
            
            logger.info(f"Block mined: {self.hash_block(new_block)}")
            logger.info(f"Block details: {new_block.to_dict()}")
            
            return new_block.to_dict()
            
        except Exception as e:
            logger.error(f"Error mining block: {str(e)}")
            return None

    def proof_of_stake(self, last_block):
        """Улучшенный Proof of Stake алгоритм"""
        total_stake = sum(self.validators.values())
        if total_stake == 0:
            return 0
            
        # Используем хеш предыдущего блока для детерминированности
        seed = int(last_block['hash'], 16)
        random.seed(seed)
        
        # Выбираем валидатора с учетом его стейка
        target = random.uniform(0, total_stake)
        current = 0
        
        for validator, stake in self.validators.items():
            current += stake
            if current >= target:
                # Создаем proof на основе стейка и предыдущего блока
                return hash(str(validator) + str(stake) + str(last_block['proof']))
        
        return 0

    def get_contract_state(self, name):
        """Возвращает текущее состояние контракта"""
        if name not in self.contract_states:
            raise ValueError(f"Contract {name} not found")
        return self.contract_states[name]['state']

    def get_contract_events(self, name):
        """Возвращает события контракта"""
        if name not in self.contract_events:
            raise ValueError(f"Contract {name} not found")
        return self.contract_events[name]

    def get_gas_info(self, transaction_id):
        """Возвращает информацию о газе для транзакции"""
        if transaction_id in self.gas_pool:
            return {
                'reserved': self.gas_pool[transaction_id],
                'used': self.gas_used.get(transaction_id, 0),
                'refunded': self.gas_refunded.get(transaction_id, 0)
            }
        return None

    def get_chain(self):
        """Возвращает полную цепочку блоков"""
        try:
            logger.info("Getting chain data...")
            logger.info(f"Chain length: {len(self.chain)}")
            logger.info(f"Chain type: {type(self.chain)}")
            
            # Логируем каждый блок в цепочке
            for i, block in enumerate(self.chain):
                logger.info(f"Block {i}: {type(block)}")
                logger.info(f"Block {i} data: {block.__json__()}")
            
            active_validators = len([v for v in self.validators if v.get('active', True)])
            logger.info(f"Active validators count: {active_validators}")
            logger.info(f"All validators: {self.validators}")
            
            network_health = self.calculate_network_health()
            logger.info(f"Network health: {network_health}")
            
            # Создаем цепочку блоков в формате словаря
            chain_dict = [block.__json__() for block in self.chain]
            logger.info(f"Chain dict type: {type(chain_dict)}")
            logger.info(f"First block dict: {chain_dict[0] if chain_dict else 'No blocks'}")
            
            chain_data = {
                'chain': chain_dict,
                'length': len(self.chain),
                'pending_transactions': self.current_transactions,
                'contract_states': self.contract_states,
                'contract_events': self.contract_events,
                'gas_pool': self.gas_pool,
                'active_validators': active_validators,
                'network_health': network_health
            }
            
            logger.info(f"Chain data type: {type(chain_data)}")
            logger.info(f"Chain data keys: {chain_data.keys()}")
            
            return json.dumps(chain_data, cls=BlockEncoder)
        except Exception as e:
            logger.error(f"Error in get_chain: {str(e)}")
            logger.exception("Full traceback:")
            raise

    def calculate_network_health(self) -> float:
        """Рассчитывает здоровье сети на основе различных метрик"""
        if not self.validators:
            return 0.0
            
        # Проверяем количество активных валидаторов
        active_validators = len([v for v in self.validators if v.get('active', True)])
        validator_health = active_validators / len(self.validators)
        
        # Проверяем количество транзакций в пуле
        tx_health = min(len(self.current_transactions) / 100, 1.0)  # Нормализуем до 100 транзакций
        
        # Проверяем длину цепи
        chain_health = min(len(self.chain) / 100, 1.0)  # Нормализуем до 100 блоков
        
        # Общее здоровье сети - среднее всех метрик
        return (validator_health + tx_health + chain_health) / 3

    def is_chain_valid(self):
        """Проверка валидности цепочки блоков"""
        if not self.chain:
            logger.error("Chain is empty")
            return False
            
        # Проверяем генезис-блок
        genesis_block = self.chain[0]
        if genesis_block.index != 0 or genesis_block.previous_hash != '0' * 64:
            logger.error("Invalid genesis block")
            return False
            
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Проверяем хеш текущего блока
            calculated_hash = current_block.calculate_hash()
            if current_block.hash != calculated_hash:
                logger.error(f"Invalid block hash at index {i}. Expected: {calculated_hash}, Got: {current_block.hash}")
                return False
                
            # Проверяем связь с предыдущим блоком
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid block link at index {i}. Expected: {previous_block.hash}, Got: {current_block.previous_hash}")
                return False
                
            # Проверяем подпись валидатора
            if not self.verify_validator_signature(current_block):
                logger.error(f"Invalid validator signature at index {i}")
                return False
                
            # Проверяем валидность транзакций
            for tx in current_block.transactions:
                if not self.validate_transaction(tx):
                    logger.error(f"Invalid transaction in block {i}")
                    return False
                    
        logger.info("Chain validation successful")
        return True

    def to_dict(self):
        """Преобразует блокчейн в словарь"""
        return {
            'chain': [block.__json__() for block in self.chain],
            'current_transactions': self.current_transactions,
            'contract_states': self.contract_states,
            'contract_events': self.contract_events,
            'gas_pool': self.gas_pool,
            'validators': self.validators
        }

    def detect_51_percent_attack(self, block: Dict) -> bool:
        """Обнаружение атаки 51%"""
        try:
            # Проверка доли стейка валидатора
            validator_stake = self.validators.get(block['validator'], 0)
            total_stake = sum(self.validators.values())
            stake_ratio = validator_stake / total_stake if total_stake > 0 else 0

            # Если валидатор контролирует более 33% стейка
            if stake_ratio > self.network_attack_threshold:
                logging.warning(f"Potential 51% attack detected: {block['validator']} has {stake_ratio:.2%} stake")
                return True

            # Проверка частоты создания блоков
            recent_blocks = [b for b in self.chain[-10:] if b['validator'] == block['validator']]
            if len(recent_blocks) > 5:  # Более 50% последних блоков
                logging.warning(f"Potential 51% attack: {block['validator']} created too many recent blocks")
                return True

            return False
        except Exception as e:
            logging.error(f"51% attack detection error: {str(e)}")
            return False

    def handle_51_percent_attack(self, block: Dict):
        """Обработка атаки 51%"""
        try:
            validator = block['validator']
            # Добавление в черный список
            self.validator_blacklist.add(validator)
            
            # Увеличение штрафа
            self.slash_conditions['double_sign'] *= 2
            
            # Перераспределение стейка
            self.redistribute_stake(validator)
            
            logging.warning(f"51% attack handled: {validator} blacklisted")
        except Exception as e:
            logging.error(f"51% attack handling error: {str(e)}")

    def detect_network_attack(self, block: Dict) -> bool:
        """Обнаружение сетевых атак"""
        try:
            # Проверка на спам транзакций
            if len(block['transactions']) > 1000:  # Слишком много транзакций
                logging.warning("Potential network attack: too many transactions")
                return True

            # Проверка на необычные паттерны в транзакциях
            if self.detect_transaction_patterns(block['transactions']):
                logging.warning("Potential network attack: suspicious transaction patterns")
                return True

            # Проверка на частые переподключения
            if self.detect_frequent_reconnections(block['validator']):
                logging.warning("Potential network attack: frequent reconnections")
                return True

            return False
        except Exception as e:
            logging.error(f"Network attack detection error: {str(e)}")
            return False

    def handle_network_attack(self, block: Dict):
        """Обработка сетевой атаки"""
        try:
            validator = block['validator']
            # Добавление в черный список
            self.validator_blacklist.add(validator)
            
            # Применение штрафа
            self.slash_validator(validator, 'network_attack')
            
            # Временное отключение валидатора
            self.temporarily_disable_validator(validator)
            
            logging.warning(f"Network attack handled: {validator} temporarily disabled")
        except Exception as e:
            logging.error(f"Network attack handling error: {str(e)}")

    def detect_transaction_patterns(self, transactions: List[Dict]) -> bool:
        """Обнаружение подозрительных паттернов в транзакциях"""
        try:
            # Проверка на одинаковые суммы
            amounts = [tx.get('amount', 0) for tx in transactions]
            if len(set(amounts)) < len(amounts) * 0.1:  # Более 90% одинаковых сумм
                return True

            # Проверка на частые транзакции между одними и теми же адресами
            pairs = [(tx.get('from'), tx.get('to')) for tx in transactions]
            if len(set(pairs)) < len(pairs) * 0.2:  # Более 80% повторяющихся пар
                return True

            return False
        except Exception as e:
            logging.error(f"Transaction pattern detection error: {str(e)}")
            return False

    def detect_frequent_reconnections(self, validator: str) -> bool:
        """Обнаружение частых переподключений"""
        try:
            # Получение истории подключений валидатора
            history = self.validators_history.get(validator, [])
            recent_events = [e for e in history if e['type'] == 'reconnect']
            
            # Если более 5 переподключений за последние 10 блоков
            if len(recent_events) > 5:
                return True

            return False
        except Exception as e:
            logging.error(f"Reconnection detection error: {str(e)}")
            return False

    def temporarily_disable_validator(self, validator: str):
        """Временное отключение валидатора"""
        try:
            # Сохранение текущего стейка
            stake = self.validators.get(validator, 0)
            
            # Удаление из активных валидаторов
            if validator in self.validators:
                del self.validators[validator]
            
            # Запись в историю
            if validator not in self.validators_history:
                self.validators_history[validator] = []
            
            self.validators_history[validator].append({
                'type': 'disable',
                'stake': stake,
                'timestamp': time.time()
            })
            
            logging.info(f"Validator {validator} temporarily disabled")
        except Exception as e:
            logging.error(f"Validator disable error: {str(e)}")

    def redistribute_stake(self, validator: str):
        """Перераспределение стейка после атаки"""
        try:
            # Получение стейка атакующего валидатора
            attacker_stake = self.validators.get(validator, 0)
            
            # Удаление валидатора
            if validator in self.validators:
                del self.validators[validator]
            
            # Распределение стейка между другими валидаторами
            total_remaining_stake = sum(self.validators.values())
            if total_remaining_stake > 0:
                for v in self.validators:
                    share = self.validators[v] / total_remaining_stake
                    self.validators[v] += attacker_stake * share
            
            logging.info(f"Stake redistributed after attack by {validator}")
        except Exception as e:
            logging.error(f"Stake redistribution error: {str(e)}")

    def select_validators(self) -> Dict[str, float]:
        """Улучшенный выбор валидаторов с защитой от атак"""
        try:
            # Фильтрация черного списка
            eligible_validators = {
                v: s for v, s in self.validators.items()
                if v not in self.validator_blacklist
            }
            
            total_stake = sum(eligible_validators.values())
            if total_stake == 0:
                return {}
            
            # Использование хеша последнего блока для детерминированности
            seed = int(self.chain[-1]['hash'], 16)
            random.seed(seed)
            
            # Выбор валидаторов с учетом защиты от атак
            selected = {}
            remaining_stake = total_stake
            
            while len(selected) < min(len(eligible_validators), self.max_validators):
                target = random.uniform(0, remaining_stake)
                current = 0
                
                for validator, stake in eligible_validators.items():
                    if validator in selected:
                        continue
                    
                    # Проверка минимальной доли стейка
                    if stake / total_stake < self.min_stake_ratio:
                        continue
                    
                    current += stake
                    if current >= target:
                        selected[validator] = stake
                        remaining_stake -= stake
                        break
            
            return selected
        except Exception as e:
            logging.error(f"Validator selection error: {str(e)}")
            return {}

    def hash_block(self, block: Any) -> str:
        """Вычисление хеша блока"""
        try:
            # Если это объект Block, используем его метод to_dict
            if hasattr(block, 'to_dict'):
                block_dict = block.to_dict()
            else:
                # Если это словарь, используем его копию
                block_dict = block.copy()
            
            # Удаляем поле hash, если оно есть
            block_dict.pop('hash', None)
            
            # Создаем строку для хеширования
            block_string = json.dumps(block_dict, sort_keys=True)
            
            # Вычисляем хеш
            return self.crypto.sha256(block_string)
        except Exception as e:
            logger.error(f"Error calculating block hash: {str(e)}")
            raise

    def deploy_contract(self, name: str, code: str, gas_limit: Optional[int] = None) -> bool:
        """Deploy a new smart contract"""
        try:
            # Validate contract name and code
            if not name or not code:
                raise ValueError("Contract name and code are required")
                
            if name in self.contract_states:
                raise ValueError(f"Contract {name} already exists")
                
            # Initialize contract state
            self.contract_states[name] = {}
            self.contract_events[name] = []
            
            # Create deployment transaction
            tx = {
                'type': 'contract_deploy',
                'name': name,
                'code': code,
                'gas_limit': gas_limit,
                'timestamp': time.time()
            }
            
            # Add transaction to pending list
            self.current_transactions.append(tx)
            
            logging.info(f"Contract {name} deployed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error deploying contract: {str(e)}")
            return False

    def process_new_block(self, block_data: Dict[str, Any]) -> bool:
        """Обработка нового блока"""
        try:
            # Добавляем задержку перед обработкой блока
            time.sleep(self.block_processing_delay)
            
            # Синхронизируем валидаторов перед обработкой блока
            if hasattr(self, 'peers') and self.peers:
                for peer in self.peers:
                    try:
                        self.sync_validators(peer)
                    except Exception as e:
                        logger.error(f"Error syncing validators with peer {peer}: {str(e)}")
            
            # Преобразуем данные блока в объект Block
            block = Block(
                index=block_data['index'],
                transactions=block_data['transactions'],
                timestamp=block_data['timestamp'],
                previous_hash=block_data['previous_hash'],
                validator=block_data['validator'],
                signature=block_data['signature'],
                epoch=block_data.get('epoch', 0)
            )
            
            # Проверяем валидность блока
            if not self.validate_block(block):
                logger.error("Invalid block received")
                return False
                
            # Проверяем подпись валидатора
            if not self.verify_validator_signature(block):
                logger.error("Invalid validator signature")
                return False
                
            # Добавляем блок в цепочку
            self.chain.append(block)
            
            # Очищаем транзакции, которые были включены в блок
            self.current_transactions = [
                tx for tx in self.current_transactions
                if tx not in block.transactions
            ]
            
            logger.info(f"New block added: {block.hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing new block: {str(e)}")
            return False

    def calculate_hash(self, block: Block) -> str:
        """Вычисляет хеш блока"""
        try:
            logger.info(f"Calculating hash for block: {block}")
            
            # Создаем строку для хеширования
            block_string = json.dumps({
                'index': block.index,
                'transactions': block.transactions,
                'timestamp': block.timestamp,
                'previous_hash': block.previous_hash,
                'validator': block.validator,
                'signature': block.signature,
                'epoch': block.epoch
            }, sort_keys=True)
            
            logger.info(f"String to hash: {block_string}")
            
            # Вычисляем хеш
            hash_result = self.crypto.sha256(block_string)
            logger.info(f"Calculated hash: {hash_result}")
            
            return hash_result
        except Exception as e:
            logger.error(f"Error calculating block hash: {str(e)}")
            logger.exception("Full traceback:")
            raise 