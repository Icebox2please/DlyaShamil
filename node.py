import socket
import threading
import json
import time
import logging
import requests
import hashlib
from blockchain import Blockchain, Block
from crypto import Crypto
from typing import Dict, List, Optional, Any
from token_standards import TokenType, TokenMetadata, ERC721, ERC1155
from token_validator import TokenValidator

logger = logging.getLogger(__name__)

class LargeNumberEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float)):
            return str(obj)  # Преобразуем большие числа в строки
        return super().default(obj)

class Node:
    def __init__(self, host: str = '127.0.0.1', port: int = 5000):
        """Инициализация узла"""
        self.host = host
        self.port = port  # Используем переданный порт
        self.peers: List[str] = []
        self.blockchain = Blockchain()
        self.crypto = Crypto()
        self.mining_thread = None
        self.server_thread = None
        self.running = False
        self.tokens = {}  # address -> token contract
        self.token_validator = TokenValidator()
        
        # Генерация ключей для узла
        self.public_key, self.private_key = self.crypto.generate_key_pair()
        logger.info(f"Node initialized on {self.host}:{self.port}")  # Используем self.port

    def start(self) -> None:
        """Запуск узла"""
        try:
            logger.info("Starting node...")
            self.running = True
            
            # Запускаем сервер в отдельном потоке
            self.server_thread = threading.Thread(target=self.start_server)
            self.server_thread.start()
            
            # Даем серверу время на запуск
            time.sleep(2)
            
            # Запускаем майнинг в отдельном потоке
            self.mining_thread = threading.Thread(target=self.mine_blocks)
            self.mining_thread.start()
            
            # Запускаем автоматическое подключение к пирам в отдельном потоке
            threading.Thread(target=self.auto_connect_peers).start()
            
            logger.info("Node started successfully")
        except Exception as e:
            logger.error(f"Error starting node: {str(e)}")
            raise

    def stop(self) -> None:
        """Остановка узла"""
        self.running = False
        if self.mining_thread:
            self.mining_thread.join()
        if self.server_thread:
            self.server_thread.join()
        logger.info("Node stopped")

    def start_server(self) -> None:
        """Запуск сервера для приема соединений от пиров"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Добавляем опцию переиспользования адреса
                s.bind((self.host, self.port))
                s.listen()
                s.settimeout(1)  # Таймаут для возможности проверки self.running
                logger.info(f"Server listening on {self.host}:{self.port}")
                
                while self.running:
                    try:
                        conn, addr = s.accept()
                        threading.Thread(target=self.handle_peer_connection, args=(conn, addr)).start()
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"Error accepting connection: {str(e)}")
        except Exception as e:
            logger.error(f"Error in server thread: {str(e)}")

    def auto_connect_peers(self) -> None:
        """Автоматическое подключение к известным пирам"""
        try:
            # Определяем, к какому пиру нужно подключиться
            if self.port == 5000:
                # Первый узел подключается ко второму
                peer = ('127.0.0.1', 5002)
            elif self.port == 5002:
                # Второй узел подключается к первому
                peer = ('127.0.0.1', 5000)
            else:
                return
                
            host, port = peer
            peer_str = f"{host}:{port}"
            
            # Проверяем, не подключены ли мы уже к этому пиру
            if peer_str not in self.peers:
                # Пытаемся подключиться с повторными попытками
                for attempt in range(10):
                    try:
                        if self.is_peer_available(host, port):
                            self.peers.append(peer_str)
                            logger.info(f"Added new peer: {peer_str}")
                            return
                        else:
                            logger.warning(f"Peer {peer_str} is not available (attempt {attempt + 1}/10)")
                            time.sleep(2)  # Уменьшаем время ожидания между попытками
                    except Exception as e:
                        logger.error(f"Error connecting to peer {peer_str}: {str(e)}")
                        time.sleep(2)
        except Exception as e:
            logger.error(f"Error in auto-connect: {str(e)}")

    def mine_blocks(self) -> None:
        """Майнинг блоков"""
        while self.running:
            try:
                # Синхронизация с пирами
                for peer in self.peers:
                    self.sync_with_peer(peer)
                    
                if self.blockchain.current_transactions:
                    logger.info("Found pending transactions, attempting to mine block...")
                    new_block = self.blockchain.mine_block(self.public_key, self.private_key)
                    
                    if new_block:
                        logger.info(f"New block mined: {new_block['index']}")
                        self.broadcast_block(new_block)
            except Exception as e:
                logger.error(f"Error in mining thread: {str(e)}")
            
            threading.Event().wait(10)

    def sync_with_peer(self, peer: str) -> None:
        """Синхронизация с пиром"""
        try:
            host, port = peer.split(':')
            if not self.is_peer_available(host, int(port)):
                logger.warning(f"Peer {peer} is not available for sync")
                return
                
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, int(port)))
                
                request = {
                    'type': 'sync_request',
                    'chain_length': len(self.blockchain.chain)
                }
                json_data = json.dumps(request, ensure_ascii=False, cls=LargeNumberEncoder)
                s.sendall(json_data.encode('utf-8'))
                
                # Увеличиваем размер буфера
                response = s.recv(4096)
                if response:
                    try:
                        data = json.loads(response.decode('utf-8'))
                        logger.debug(f"Received sync response: {data}")
                        if data['type'] == 'sync_response':
                            for block in data['blocks']:
                                if isinstance(block, dict) and block['index'] > len(self.blockchain.chain):
                                    self.process_new_block(block)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {str(e)}")
                        logger.error(f"Raw response: {response.decode('utf-8')}")
        except Exception as e:
            logger.error(f"Error syncing with peer {peer}: {str(e)}")

    def broadcast_block(self, block: Dict[str, Any]) -> None:
        """Рассылка нового блока всем пирам"""
        for peer in self.peers:
            try:
                self.send_block(peer, block)
            except Exception as e:
                logger.error(f"Error broadcasting block to {peer}: {str(e)}")

    def send_block(self, peer: str, block: Dict[str, Any]) -> None:
        """Отправка блока пиру"""
        try:
            host, port = peer.split(':')
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(10)  # Увеличиваем таймаут для ожидания ответа
                s.connect((host, int(port)))
                
                if hasattr(block, '__json__'):
                    block_dict = block.__json__()
                else:
                    block_dict = block
                
                message = {
                    'type': 'new_block',
                    'block': block_dict
                }
                json_data = json.dumps(message, ensure_ascii=False, cls=LargeNumberEncoder)
                logger.info(f"Sending block to {peer}: index={block_dict.get('index')}, hash={block_dict.get('hash')}")
                s.sendall(json_data.encode('utf-8'))
                
                # Ждем подтверждения получения
                response = s.recv(4096)
                if response:
                    try:
                        data = json.loads(response.decode('utf-8'))
                        if data['type'] == 'block_received':
                            logger.info(f"Block confirmation from {peer}: success={data['success']}, hash={data['block_hash']}")
                        else:
                            logger.warning(f"Unexpected response type from {peer}: {data['type']}")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in block confirmation: {str(e)}")
        except Exception as e:
            logger.error(f"Error sending block to {peer}: {str(e)}")
            raise

    def handle_peer_connection(self, conn: socket.socket, addr: tuple) -> None:
        """Обработка подключения пира"""
        try:
            # Увеличиваем размер буфера
            data = conn.recv(4096)
            if data:
                try:
                    message = json.loads(data.decode('utf-8'))
                    logger.debug(f"Received message: {message}")
                    
                    if message['type'] == 'new_block':
                        block = message['block']
                        logger.info(f"Received new block from {addr}: index={block.get('index')}, hash={block.get('hash')}")
                        if isinstance(block, dict):
                            success = self.process_new_block(block)
                            # Отправляем подтверждение
                            response = {
                                'type': 'block_received',
                                'success': success,
                                'block_hash': block.get('hash')
                            }
                            json_data = json.dumps(response, ensure_ascii=False, cls=LargeNumberEncoder)
                            conn.sendall(json_data.encode('utf-8'))
                        else:
                            logger.error(f"Received invalid block format: {type(block)}")
                    elif message['type'] == 'sync_request':
                        blocks_to_send = [block.__json__() if hasattr(block, '__json__') else block 
                                        for block in self.blockchain.chain[message['chain_length']:]]
                        response = {
                            'type': 'sync_response',
                            'blocks': blocks_to_send
                        }
                        json_data = json.dumps(response, ensure_ascii=False, cls=LargeNumberEncoder)
                        conn.sendall(json_data.encode('utf-8'))
                    elif message['type'] == 'ping':
                        # Отвечаем на пинг-запрос
                        response = {'type': 'pong'}
                        json_data = json.dumps(response, ensure_ascii=False, cls=LargeNumberEncoder)
                        conn.sendall(json_data.encode('utf-8'))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    logger.error(f"Raw data: {data.decode('utf-8')}")
        except Exception as e:
            logger.error(f"Error handling peer connection: {str(e)}")
        finally:
            conn.close()

    def process_new_block(self, block: Dict[str, Any]) -> bool:
        """Обработка нового блока"""
        try:
            logger.info(f"Processing new block: index={block.get('index')}, hash={block.get('hash')}")
            # Синхронизируем валидаторов перед обработкой блока
            for peer in self.peers:
                try:
                    logger.info(f"Syncing validators with peer {peer} before processing block")
                    self.sync_validators_with_peer(peer)
                except Exception as e:
                    logger.error(f"Error syncing validators with peer {peer}: {str(e)}")
            
            return self.blockchain.process_new_block(block)
        except Exception as e:
            logger.error(f"Error processing new block: {str(e)}")
            return False

    def validate_block(self, block: Dict[str, Any]) -> bool:
        """Валидация блока"""
        try:
            return self.blockchain.validate_block(block)
        except Exception as e:
            logger.error(f"Error validating block: {str(e)}")
            return False

    def add_transaction(self, sender: str, recipient: str, amount: float) -> bool:
        """Добавление новой транзакции"""
        try:
            return self.blockchain.add_transaction(sender, recipient, amount)
        except Exception as e:
            logger.error(f"Error adding transaction: {str(e)}")
            return False

    def deploy_contract(self, name: str, code: str) -> bool:
        """Деплой смарт-контракта"""
        try:
            return self.blockchain.deploy_contract(name, code)
        except Exception as e:
            logger.error(f"Error deploying contract: {str(e)}")
            return False

    def execute_contract(self, contract: str, function: str, args: List[Any]) -> Any:
        """Выполнение смарт-контракта"""
        try:
            return self.blockchain.execute_contract(contract, function, args)
        except Exception as e:
            logger.error(f"Error executing contract: {str(e)}")
            raise

    def add_validator(self, address: str, stake: float) -> bool:
        """Добавление валидатора"""
        try:
            return self.blockchain.add_validator(address, stake)
        except Exception as e:
            logger.error(f"Error adding validator: {str(e)}")
            return False

    def get_chain(self) -> Dict[str, Any]:
        """Получение состояния блокчейна"""
        try:
            # Преобразуем блоки в словари
            chain_dict = []
            for block in self.blockchain.chain:
                if isinstance(block, Block):
                    block_dict = block.to_dict()
                else:
                    block_dict = block.copy()
                
                if 'hash' not in block_dict:
                    block_dict['hash'] = self.blockchain.hash_block(block_dict)
                
                chain_dict.append(block_dict)
            
            chain_data = {
                'chain': chain_dict,
                'pending_transactions': self.blockchain.current_transactions,
                'validators': self.blockchain.validators,
                'contract_states': self.blockchain.contract_states,
                'contract_events': self.blockchain.contract_events
            }
            return chain_data
        except Exception as e:
            logger.error(f"Error getting chain data: {str(e)}")
            raise

    def add_peer(self, host: str, port: int) -> bool:
        """Добавление пира"""
        try:
            peer = f"{host}:{port}"
            if peer not in self.peers and peer != f"{self.host}:{self.port}":
                # Проверяем доступность пира перед добавлением
                if self.is_peer_available(host, port):
                    self.peers.append(peer)
                    logger.info(f"Added new peer: {peer}")
                    return True
                else:
                    logger.warning(f"Peer {peer} is not available")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error adding peer {host}:{port}: {str(e)}")
            return False

    def is_peer_available(self, host: str, port: int) -> bool:
        """Проверка доступности пира"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  # Увеличиваем таймаут до 5 секунд
                s.connect((host, port))
                # Отправляем пинг-запрос
                s.sendall(json.dumps({'type': 'ping'}, ensure_ascii=False, cls=LargeNumberEncoder).encode())
                # Ждем ответ
                response = s.recv(4096)
                if response:
                    try:
                        data = json.loads(response.decode())
                        return data.get('type') == 'pong'
                    except json.JSONDecodeError:
                        return False
                return False
        except Exception as e:
            logger.error(f"Error checking peer {host}:{port}: {str(e)}")
            return False

    def sync_validators_with_peer(self, peer: str) -> None:
        """Синхронизация валидаторов с пиром"""
        try:
            host, port = peer.split(':')
            if not self.is_peer_available(host, int(port)):
                logger.warning(f"Peer {peer} is not available for validator sync")
                return
                
            # Получаем валидаторов через HTTP API
            response = requests.get(f"http://{host}:{int(port)+1}/api/validators")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    peer_validators = data.get('validators', {})
                    logger.info(f"Received validators from peer {peer}: {peer_validators}")
                    # Обновляем локальный список валидаторов
                    for address, stake in peer_validators.items():
                        if stake > 0:
                            if address not in self.blockchain.validators or self.blockchain.validators[address] != stake:
                                self.blockchain.validators[address] = stake
                                logger.info(f"Updated validator {address} with stake {stake}")
                else:
                    logger.error(f"Failed to get validators from peer {peer}: {data.get('error')}")
            else:
                logger.error(f"Failed to get validators from peer {peer}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error syncing validators with peer {peer}: {str(e)}")

    def create_token(self, token_type: str, data: dict) -> dict:
        """Создание нового токена"""
        try:
            # Валидация входных данных
            if not data.get('name') or not data.get('symbol'):
                raise ValueError("Name and symbol are required")

            # Создание метаданных
            metadata = TokenMetadata(
                name=data['name'],
                description=data.get('description', ''),
                image=data.get('metadata', {}).get('image', ''),
                attributes=data.get('metadata', {}).get('attributes', {})
            )

            # Создание контракта в зависимости от типа
            if token_type == 'erc721':
                contract = ERC721(data['name'], data['symbol'])
            elif token_type == 'erc1155':
                contract = ERC1155(data['name'])
            else:
                raise ValueError(f"Unsupported token type: {token_type}")

            # Генерация адреса для токена
            address = self._generate_address()

            # Создание транзакции для деплоя контракта
            tx = {
                'type': 'contract_deploy',
                'name': address,
                'code': {
                    'type': token_type,
                    'name': data['name'],
                    'symbol': data['symbol'],
                    'metadata': metadata.to_json()
                },
                'timestamp': time.time()
            }

            # Добавление транзакции в блокчейн
            self.blockchain.current_transactions.append(tx)

            # Сохранение контракта локально
            self.tokens[address] = contract

            return {
                'address': address,
                'type': token_type,
                'name': data['name'],
                'symbol': data['symbol'],
                'metadata': metadata.to_json()
            }
        except Exception as e:
            logger.error(f"Error creating token: {str(e)}")
            raise

    def get_tokens(self) -> list:
        """Получение списка всех токенов"""
        try:
            # Получаем состояния контрактов из блокчейна
            chain_data = self.blockchain.get_chain()
            contract_states = chain_data.get('contract_states', {})

            tokens = []
            for address, contract in self.tokens.items():
                # Получаем состояние контракта из блокчейна
                state = contract_states.get(address, {})
                
                tokens.append({
                    'address': address,
                    'type': 'erc721' if isinstance(contract, ERC721) else 'erc1155',
                    'name': contract.name,
                    'symbol': getattr(contract, 'symbol', ''),
                    'totalSupply': contract.total_supply if hasattr(contract, 'total_supply') else 0,
                    'state': state
                })
            return tokens
        except Exception as e:
            logger.error(f"Error getting tokens: {str(e)}")
            raise

    def get_token(self, token_id: str) -> Optional[dict]:
        """Получение информации о токене"""
        try:
            contract = self.tokens.get(token_id)
            if not contract:
                return None

            # Получаем состояние контракта из блокчейна
            chain_data = self.blockchain.get_chain()
            contract_states = chain_data.get('contract_states', {})
            state = contract_states.get(token_id, {})

            return {
                'address': token_id,
                'type': 'erc721' if isinstance(contract, ERC721) else 'erc1155',
                'name': contract.name,
                'symbol': getattr(contract, 'symbol', ''),
                'totalSupply': contract.total_supply if hasattr(contract, 'total_supply') else 0,
                'metadata': contract.metadata.to_json() if hasattr(contract, 'metadata') else None,
                'state': state
            }
        except Exception as e:
            logger.error(f"Error getting token: {str(e)}")
            raise

    def transfer_token(self, token_id: str, to: str, amount: Optional[int] = None) -> bool:
        """Передача токена"""
        try:
            contract = self.tokens.get(token_id)
            if not contract:
                raise ValueError("Token not found")

            if not self.token_validator.validate_address(to):
                raise ValueError("Invalid recipient address")

            # Создание транзакции для передачи токена
            tx = {
                'type': 'contract_execution',
                'contract': token_id,
                'function': 'transfer',
                'args': {
                    'from': contract.owner,
                    'to': to,
                    'token_id': token_id,
                    'amount': amount
                },
                'timestamp': time.time()
            }

            # Добавление транзакции в блокчейн
            self.blockchain.current_transactions.append(tx)

            # Выполнение передачи локально
            if isinstance(contract, ERC721):
                return contract.transfer(contract.owner, to, token_id, contract.owner)
            elif isinstance(contract, ERC1155):
                if amount is None:
                    raise ValueError("Amount is required for ERC1155 tokens")
                return contract.transfer(contract.owner, to, token_id, amount, contract.owner)
            else:
                raise ValueError("Unsupported token type")

        except Exception as e:
            logger.error(f"Error transferring token: {str(e)}")
            raise

    def get_token_metadata(self, token_id: str) -> Optional[dict]:
        """Получение метаданных токена"""
        try:
            contract = self.tokens.get(token_id)
            if not contract or not hasattr(contract, 'metadata'):
                return None

            # Получаем состояние контракта из блокчейна
            chain_data = self.blockchain.get_chain()
            contract_states = chain_data.get('contract_states', {})
            state = contract_states.get(token_id, {})

            metadata = contract.metadata.to_json()
            metadata['state'] = state
            return metadata
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            raise

    def _generate_address(self) -> str:
        """Генерация уникального адреса для токена"""
        return f"0x{hashlib.sha256(str(time.time()).encode()).hexdigest()[:40]}" 