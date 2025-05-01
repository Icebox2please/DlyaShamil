import argparse
import logging
import socket
from flask import Flask, jsonify, request, render_template
from json_encoder import CustomJSONEncoder
from node import Node
from blockchain import Blockchain
from metrics import MetricsCollector
from dsl import SmartContract, DataType, Parameter, Function, ContractState
from contract_parser import parse_contract
import json
import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
from blockchain import Block
import functools
from datetime import datetime, timedelta
import psutil
from system_metrics_collector import SystemMetricsCollector
from vm import VirtualMachine

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создаем отдельный логгер для метрик производительности
metrics_logger = logging.getLogger('performance_metrics')
metrics_logger.setLevel(logging.INFO)
metrics_handler = logging.FileHandler('performance_metrics.log')
metrics_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
metrics_logger.addHandler(metrics_handler)

# Декоратор для измерения времени выполнения
def log_performance(metric_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Логируем метрику
            metrics_logger.info(f"{metric_name} - Execution time: {execution_time:.4f} seconds")
            
            return result
        return wrapper
    return decorator

# Декоратор для измерения времени выполнения API запросов
def log_api_performance(metric_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Логируем метрику с дополнительной информацией
            metrics_logger.info(
                f"API {metric_name} - "
                f"Execution time: {execution_time:.4f} seconds, "
                f"Method: {request.method}, "
                f"Path: {request.path}"
            )
            
            return result
        return wrapper
    return decorator

# Определение путей
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Создание папок, если они не существуют
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Templates directory: {TEMPLATES_DIR}")
logger.info(f"Static directory: {STATIC_DIR}")

app = Flask(__name__,
            template_folder=TEMPLATES_DIR,
            static_folder=STATIC_DIR)
app.json_encoder = CustomJSONEncoder
node = None

# Инициализация сборщика метрик
metrics_collector = MetricsCollector()

# Инициализация сборщика системных метрик
system_metrics_collector = SystemMetricsCollector()

def is_port_in_use(port: int) -> bool:
    """Проверка занятости порта"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except Exception as e:
        logger.error(f"Error checking port {port}: {str(e)}")
        return True

def find_available_port(start_port: int = 5000, max_attempts: int = 100) -> int:
    """Поиск доступного порта"""
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        if not is_port_in_use(port):
            logger.info(f"Found available port: {port}")
            return port
        logger.warning(f"Port {port} is in use, trying next port...")
        port += 1
        attempts += 1
    
    raise RuntimeError(f"Could not find available port after {max_attempts} attempts")

def is_peer_available(host: str, port: int) -> bool:
    """Проверка доступности пира"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)  # Увеличиваем таймаут до 2 секунд
            return s.connect_ex((host, port)) == 0
    except Exception as e:
        logger.error(f"Error checking peer {host}:{port}: {str(e)}")
        return False

def connect_to_peer_with_retry(host: str, port: int, max_retries: int = 5, retry_delay: int = 5) -> bool:
    """Подключение к пиру с повторными попытками"""
    for attempt in range(max_retries):
        try:
            if is_peer_available(host, port):
                if node.add_peer(host, port):
                    logger.info(f"Successfully connected to peer {host}:{port}")
                    return True
                else:
                    logger.warning(f"Failed to add peer {host}:{port}")
            else:
                logger.warning(f"Peer {host}:{port} is not available (attempt {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying connection to {host}:{port} in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Error connecting to peer {host}:{port}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    
    return False

def connect_to_known_peers(port: int) -> None:
    """Автоматическое подключение к известным пирам"""
    try:
        # Список известных пиров
        known_peers = [
            ('127.0.0.1', 5000),  # Первый узел
            ('127.0.0.1', 5002)   # Второй узел
        ]
        
        for host, peer_port in known_peers:
            if peer_port != port:  # Не подключаемся к себе
                # Проверяем, не подключены ли мы уже к этому пиру
                is_connected = False
                for peer in node.peers:
                    if isinstance(peer, dict) and peer.get('host') == host and peer.get('port') == peer_port:
                        is_connected = True
                        break
                
                if not is_connected:
                    connect_to_peer_with_retry(host, peer_port)
    except Exception as e:
        logger.error(f"Error in connect_to_known_peers: {str(e)}")
        logger.error(f"Current peers: {node.peers}")

def start_peer_connection_thread(port: int) -> None:
    """Запуск потока для периодической проверки и подключения к пирам"""
    def peer_connection_loop():
        while True:
            try:
                connect_to_known_peers(port)
                time.sleep(10)  # Проверяем каждые 10 секунд
            except Exception as e:
                logger.error(f"Error in peer connection loop: {str(e)}")
                time.sleep(10)
    
    import threading
    thread = threading.Thread(target=peer_connection_loop, daemon=True)
    thread.start()

@app.before_request
def log_request_info():
    """Логирование входящих запросов"""
    logger.info(f"Request: {request.method} {request.url}")
    if request.is_json:
        logger.info(f"Request body: {request.json}")

@app.after_request
def log_response_info(response):
    """Логирование исходящих ответов"""
    logger.info(f"Response: {response.status}")
    return response

@app.errorhandler(Exception)
def handle_error(error):
    """Обработка ошибок"""
    logger.error(f"Error: {str(error)}")
    return jsonify({
        'error': str(error),
        'status': 'error'
    }), 500

@app.route('/')
def index():
    return render_template('home.html', active_page='home')

@app.route('/contract')
def contract():
    return render_template('contract.html')

@app.route('/blocks')
def blocks():
    return render_template('blocks.html', active_page='blocks')

@app.route('/transactions')
def transactions():
    return render_template('transactions.html', active_page='transactions')

@app.route('/validators')
def validators():
    return render_template('validators.html', active_page='validators')

@app.route('/contracts')
def contracts():
    return render_template('contracts.html', active_page='contracts')

@log_api_performance("get_chain")
@app.route('/api/chain', methods=['GET'])
def get_chain():
    """Получение состояния блокчейна"""
    try:
        start_time = time.time()  # Добавляем измерение времени
        chain_data = node.get_chain()
        
        # Преобразуем блоки в словари
        chain_with_hashes = []
        for block in chain_data['chain']:
            if isinstance(block, Block):
                block_dict = block.__json__()
            else:
                block_dict = block.copy()
            
            if 'hash' not in block_dict:
                block_dict['hash'] = node.blockchain.hash_block(block_dict)
            
            chain_with_hashes.append(block_dict)
        
        # Собираем метрики блокчейна
        blockchain_metrics = {
            "block_time": time.time() - start_time,
            "transactions_count": len(chain_data['chain'][-1].transactions) if chain_data['chain'] else 0,
            "block_size": len(json.dumps(chain_data['chain'][-1].__dict__)) if chain_data['chain'] else 0
        }
        metrics_collector.collect_blockchain_metrics(blockchain_metrics)
        
        return jsonify({
            'chain': chain_with_hashes,
            'pending_transactions': chain_data['pending_transactions'],
            'validators': chain_data['validators'],
            'contract_states': chain_data.get('contract_states', {}),
            'contract_events': chain_data.get('contract_events', {}),
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting chain: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/node/info', methods=['GET'])
def get_node_info():
    """Получение информации о ноде"""
    try:
        start_time = time.time()  # Добавляем измерение времени
        # Получаем информацию о блокчейне
        chain_info = node.get_chain()
        
        # Формируем ответ
        response = {
            'public_key': node.public_key if hasattr(node, 'public_key') else 'Not available',
            'host': node.host,
            'port': node.port,
            'peers': node.peers,
            'chain_length': len(chain_info['chain']),
            'pending_transactions': len(chain_info['pending_transactions']),
            'validators': len(chain_info.get('validators', {})),
            'network_health': chain_info.get('network_health', 0)
        }
        
        # Собираем метрики узла
        node_metrics = {
            "status": "up",
            "response_time": time.time() - start_time,
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "cpu_usage": psutil.Process().cpu_percent()
        }
        metrics_collector.collect_node_metrics(node.public_key, node_metrics)
        
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error getting node info: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@log_api_performance("add_validator")
@app.route('/api/validators/add', methods=['POST'])
def add_validator():
    try:
        data = request.get_json()
        address = data.get('address')
        stake = data.get('stake')
        
        if not address or stake is None:
            return jsonify({
                'status': 'error',
                'error': 'Missing required fields: address and stake'
            }), 400
            
        # Добавляем валидатора в блокчейн
        node.blockchain.add_validator(address, stake)
        
        return jsonify({
            'status': 'success',
            'message': 'Validator added successfully'
        })
    except Exception as e:
        logger.error(f"Error adding validator: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@log_performance("node_startup")
def start_node(port: int = 5000) -> None:
    """Запуск узла"""
    try:
        # Проверяем, доступен ли порт для Node
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    s.close()
                break
            except OSError:
                logger.warning(f"Port {port} is in use, trying next port...")
                port += 1

        # Находим порт для Flask (следующий после порта Node)
        flask_port = port + 1
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', flask_port))
                    s.close()
                break
            except OSError:
                logger.warning(f"Port {flask_port} is in use, trying next port...")
                flask_port += 1

        logger.info(f"Found available port: {port}")
        
        # Создаем и запускаем узел
        global node
        node = Node(port=port)
        node.start()
        
        # Запускаем Flask приложение
        logger.info(f"Starting Flask application on port {flask_port}")
        app.run(host='127.0.0.1', port=flask_port, debug=False)
    except Exception as e:
        logger.error(f"Error starting node: {str(e)}")
        raise

@log_performance("node_process")
def run_node(port: int) -> None:
    """Функция для запуска узла в отдельном процессе"""
    try:
        # Создаем новый экземпляр Flask для каждого узла
        from flask import Flask
        node_app = Flask(__name__,
                        template_folder=TEMPLATES_DIR,
                        static_folder=STATIC_DIR)
        
        # Создаем и запускаем узел
        global local_node
        local_node = Node(port=port)
        local_node.start()
        
        # Добавляем маршрут для получения информации об узле
        @node_app.route('/api/node/info')
        def get_node_info_wrapper():
            try:
                # Получаем информацию о блокчейне
                chain_info = local_node.get_chain()
                
                # Формируем ответ
                response = {
                    'public_key': local_node.public_key if hasattr(local_node, 'public_key') else 'Not available',
                    'host': local_node.host,
                    'port': local_node.port,
                    'peers': local_node.peers,
                    'chain_length': len(chain_info['chain']),
                    'pending_transactions': len(chain_info['pending_transactions']),
                    'validators': len(chain_info.get('validators', {})),
                    'network_health': chain_info.get('network_health', 0)
                }
                
                return jsonify(response), 200
            except Exception as e:
                logger.error(f"Error getting node info: {str(e)}")
                return jsonify({'error': str(e)}), 500

        # Добавляем маршрут для получения списка валидаторов
        @node_app.route('/api/validators', methods=['GET'])
        def get_validators_wrapper():
            try:
                if not local_node:
                    logger.error("Local node not initialized")
                    return jsonify({
                        'error': 'Node not initialized',
                        'status': 'error'
                    }), 500
                    
                # Получаем список валидаторов
                validators = local_node.blockchain.validators
                logger.info(f"Retrieved {len(validators)} validators")
                
                return jsonify({
                    'validators': validators,
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"Error getting validators: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        # Добавляем маршрут для добавления валидатора
        @node_app.route('/api/validators/add', methods=['POST'])
        def add_validator_wrapper():
            try:
                data = request.get_json()
                address = data.get('address')
                stake = data.get('stake')
                
                if not address or stake is None:
                    return jsonify({
                        'status': 'error',
                        'error': 'Missing required fields: address and stake'
                    }), 400
                    
                # Добавляем валидатора в блокчейн
                local_node.blockchain.add_validator(address, stake)
                
                return jsonify({
                    'status': 'success',
                    'message': 'Validator added successfully'
                })
            except Exception as e:
                logger.error(f"Error adding validator: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
        
        # Копируем все маршруты из основного приложения
        for rule in app.url_map.iter_rules():
            if rule.endpoint != 'static' and rule.endpoint != 'get_node_info' and rule.endpoint != 'add_validator':
                if rule.endpoint == 'get_chain':
                    # Специальная обработка для маршрута get_chain
                    def get_chain_wrapper():
                        try:
                            chain_data = local_node.get_chain()
                            chain_with_hashes = []
                            for block in chain_data['chain']:
                                block_with_hash = block.copy()
                                block_with_hash['hash'] = local_node.blockchain.hash_block(block)
                                chain_with_hashes.append(block_with_hash)
                            
                            return jsonify({
                                'chain': chain_with_hashes,
                                'pending_transactions': chain_data['pending_transactions'],
                                'validators': chain_data['validators'],
                                'contract_states': chain_data.get('contract_states', {}),
                                'contract_events': chain_data.get('contract_events', {}),
                                'status': 'success'
                            })
                        except Exception as e:
                            logger.error(f"Error getting chain: {str(e)}")
                            return jsonify({
                                'error': str(e),
                                'status': 'error'
                            }), 500
                    node_app.add_url_rule(rule.rule, rule.endpoint, get_chain_wrapper)
                elif rule.endpoint == 'handle_contracts':
                    # Специальная обработка для маршрута contracts
                    def handle_contracts_wrapper():
                        if request.method == 'GET':
                            try:
                                contracts = local_node.blockchain.contract_states
                                return jsonify({
                                    'status': 'success',
                                    'contracts': contracts
                                })
                            except Exception as e:
                                logger.error(f"Error getting contracts: {str(e)}")
                                return jsonify({
                                    'status': 'error',
                                    'error': str(e)
                                }), 500
                        elif request.method == 'POST':
                            try:
                                data = request.get_json()
                                if not data:
                                    return jsonify({
                                        'status': 'error',
                                        'error': 'No data provided'
                                    }), 400
                                
                                name = data.get('name')
                                code = data.get('code')
                                
                                if not name or not code:
                                    return jsonify({
                                        'status': 'error',
                                        'error': 'Missing required fields: name and code'
                                    }), 400
                                
                                # Парсим контракт
                                contract = parse_contract(code)
                                
                                # Деплоим контракт в блокчейн
                                success = local_node.blockchain.deploy_contract(name, code)
                                
                                if success:
                                    return jsonify({
                                        'status': 'success',
                                        'message': 'Contract deployed successfully',
                                        'contract': {
                                            'name': contract.name,
                                            'state': contract.state.__json__(),
                                            'functions': list(contract.functions.keys())
                                        }
                                    })
                                else:
                                    return jsonify({
                                        'status': 'error',
                                        'error': 'Failed to deploy contract'
                                    }), 500
                            except Exception as e:
                                logger.error(f"Error deploying contract: {str(e)}")
                                return jsonify({
                                    'status': 'error',
                                    'error': str(e)
                                }), 500
                    node_app.add_url_rule(rule.rule, rule.endpoint, handle_contracts_wrapper, methods=['GET', 'POST'])
                else:
                    node_app.add_url_rule(rule.rule, rule.endpoint, app.view_functions[rule.endpoint])
        
        # Запускаем Flask приложение
        flask_port = port + 1
        node_app.run(host='127.0.0.1', port=flask_port, debug=False)
    except Exception as e:
        logger.error(f"Error in node process {port}: {str(e)}")

def start_both_nodes() -> None:
    """Запуск обоих узлов"""
    try:
        # Создаем два процесса для запуска узлов
        import multiprocessing
        
        # Запускаем первый узел (порт 5000)
        process1 = multiprocessing.Process(target=run_node, args=(5000,))
        process1.start()
        
        # Даем время на запуск первого узла
        time.sleep(5)
        
        # Запускаем второй узел (порт 5002)
        process2 = multiprocessing.Process(target=run_node, args=(5002,))
        process2.start()
        
        # Ждем завершения процессов
        process1.join()
        process2.join()
        
    except Exception as e:
        logger.error(f"Error starting nodes: {str(e)}")
        raise

@app.route('/api/validators', methods=['GET'])
def get_validators():
    """Получение списка валидаторов"""
    try:
        # Пытаемся получить узел из глобальной области или локального процесса
        current_node = None
        if 'node' in globals() and node is not None:
            current_node = node
        elif 'local_node' in globals() and local_node is not None:
            current_node = local_node
            
        if not current_node:
            logger.error("No node instance available")
            return jsonify({
                'error': 'Node not initialized',
                'status': 'error'
            }), 500
            
        # Получаем список валидаторов
        validators = current_node.blockchain.validators
        logger.info(f"Retrieved {len(validators)} validators")
        
        return jsonify({
            'validators': validators,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting validators: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/metrics')
def metrics():
    return render_template('metrics.html', active_page='metrics')

@app.route('/api/metrics')
def get_metrics():
    """Получение метрик производительности"""
    try:
        # Получаем метрики за последние 24 часа
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        
        report = metrics_collector.generate_performance_report(start_time, end_time)
        
        return jsonify({
            'metrics': report,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/performance')
def performance():
    return render_template('performance.html', active_page='performance')

@app.route('/system-metrics')
def system_metrics():
    return render_template('system_metrics.html', active_page='system_metrics')

@app.route('/api/system-metrics')
def get_system_metrics():
    """Получение системных метрик"""
    try:
        # Собираем текущие метрики
        system_metrics_collector.collect_metrics()
        
        # Получаем отчет за последний час
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        report = system_metrics_collector.get_metrics_report(start_time, end_time)
        
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Token API endpoints
@app.route('/api/tokens', methods=['GET'])
@log_api_performance("get_tokens")
def get_tokens():
    """Получение списка всех токенов"""
    try:
        tokens = node.get_tokens()
        return jsonify({
            'status': 'success',
            'tokens': tokens
        })
    except Exception as e:
        logger.error(f"Error getting tokens: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/tokens/<token_type>', methods=['POST'])
@log_api_performance("create_token")
def create_token(token_type):
    """Создание нового токена"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400

        result = node.create_token(token_type, data)
        return jsonify({
            'status': 'success',
            'token': result
        })
    except Exception as e:
        logger.error(f"Error creating token: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/tokens/<token_id>', methods=['GET'])
@log_api_performance("get_token")
def get_token(token_id):
    """Получение информации о токене"""
    try:
        token = node.get_token(token_id)
        if not token:
            return jsonify({
                'status': 'error',
                'message': 'Token not found'
            }), 404
        return jsonify({
            'status': 'success',
            'token': token
        })
    except Exception as e:
        logger.error(f"Error getting token: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/tokens/<token_id>/transfer', methods=['POST'])
@log_api_performance("transfer_token")
def transfer_token(token_id):
    """Передача токена"""
    try:
        data = request.get_json()
        if not data or 'to' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid transfer data'
            }), 400

        result = node.transfer_token(token_id, data['to'], data.get('amount'))
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        logger.error(f"Error transferring token: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/tokens/<token_id>/metadata', methods=['GET'])
@log_api_performance("get_token_metadata")
def get_token_metadata(token_id):
    """Получение метаданных токена"""
    try:
        metadata = node.get_token_metadata(token_id)
        if not metadata:
            return jsonify({
                'status': 'error',
                'message': 'Token metadata not found'
            }), 404
        return jsonify({
            'status': 'success',
            'metadata': metadata
        })
    except Exception as e:
        logger.error(f"Error getting token metadata: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/contracts', methods=['GET', 'POST'])
@log_api_performance("contracts")
def handle_contracts():
    """Обработка GET и POST запросов для контрактов"""
    if request.method == 'GET':
        try:
            # Пытаемся получить узел из глобальной области или локального процесса
            current_node = None
            if 'node' in globals() and node is not None:
                current_node = node
            elif 'local_node' in globals() and local_node is not None:
                current_node = local_node
                
            if not current_node:
                logger.error("No node instance available")
                return jsonify({
                    'error': 'Node not initialized',
                    'status': 'error'
                }), 500
                
            contracts = current_node.blockchain.contract_states
            return jsonify({
                'status': 'success',
                'contracts': contracts
            })
        except Exception as e:
            logger.error(f"Error getting contracts: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
    elif request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'status': 'error',
                    'error': 'No data provided'
                }), 400
                
            name = data.get('name')
            code = data.get('code')
            
            if not name or not code:
                return jsonify({
                    'status': 'error',
                    'error': 'Missing required fields: name and code'
                }), 400
                
            # Пытаемся получить узел из глобальной области или локального процесса
            current_node = None
            if 'node' in globals() and node is not None:
                current_node = node
            elif 'local_node' in globals() and local_node is not None:
                current_node = local_node
                
            if not current_node:
                logger.error("No node instance available")
                return jsonify({
                    'error': 'Node not initialized',
                    'status': 'error'
                }), 500
                
            # Парсим контракт
            contract = parse_contract(code)
            
            # Деплоим контракт в блокчейн
            success = current_node.blockchain.deploy_contract(name, code)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'Contract deployed successfully',
                    'contract': {
                        'name': contract.name,
                        'state': contract.state.__json__(),
                        'functions': list(contract.functions.keys())
                    }
                })
            else:
                return jsonify({
                    'status': 'error',
                    'error': 'Failed to deploy contract'
                }), 500
        except Exception as e:
            logger.error(f"Error deploying contract: {str(e)}")
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500

@app.route('/api/contracts/<contract_name>', methods=['GET'])
@log_api_performance("get_contract")
def get_contract(contract_name):
    """Получение информации о контракте"""
    try:
        state = node.blockchain.get_contract_state(contract_name)
        events = node.blockchain.get_contract_events(contract_name)
        
        if state is None:
            return jsonify({
                'status': 'error',
                'error': 'Contract not found'
            }), 404
            
        return jsonify({
            'status': 'success',
            'contract': {
                'name': contract_name,
                'state': state,
                'events': events
            }
        })
    except Exception as e:
        logger.error(f"Error getting contract: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/contracts/<contract_name>/execute', methods=['POST'])
@log_api_performance("execute_contract")
def execute_contract(contract_name):
    """Выполнение функции контракта"""
    try:
        data = request.get_json()
        function = data.get('function')
        args = data.get('args', [])
        
        if not function:
            return jsonify({
                'status': 'error',
                'error': 'Missing required field: function'
            }), 400
            
        # Выполняем функцию контракта
        result = node.blockchain.execute_contract(contract_name, function, args)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        logger.error(f"Error executing contract: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Blockchain Node')
    parser.add_argument('--single', action='store_true', help='Run single node')
    parser.add_argument('--port', type=int, help='Port to run the single node on')
    args = parser.parse_args()

    try:
        if args.single:
            # Запускаем один узел (для обратной совместимости)
            port = args.port if args.port else 5000
            start_node(port)
        else:
            # По умолчанию запускаем оба узла
            start_both_nodes()
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        sys.exit(1) 