import logging

class SmartContractDSL:
    def __init__(self):
        self.contracts = {}
        self.logger = logging.getLogger(__name__)
        
        # Стоимость операций в газе
        self.GAS_COSTS = {
            'ADD': 3,
            'MUL': 5,
            'DIV': 5,
            'MOD': 5,
            'EXP': 10,
            'SLOAD': 200,      # Чтение из хранилища
            'SSTORE': 20000,   # Запись в хранилище
            'CONTRACT_CALL': 700,
            'EVENT_EMIT': 375,
            'REQUIRE': 10
        }
        
        # Лимиты выполнения
        self.EXECUTION_LIMITS = {
            'MAX_GAS_PER_TX': 1000000,
            'MAX_CALL_DEPTH': 1024,
            'MAX_STACK_SIZE': 1024,
            'MAX_MEMORY': 32 * 1024 * 1024  # 32MB
        }
        
        self.logger.info("SmartContractDSL initialized")

    def deploy(self, name, code, gas_limit=None):
        """Деплой нового контракта с лимитом газа"""
        try:
            if gas_limit is None:
                gas_limit = self.EXECUTION_LIMITS['MAX_GAS_PER_TX']
                
            # Парсинг и валидация кода контракта
            contract = self._parse_contract(code)
            contract['gas_limit'] = gas_limit
            self.contracts[name] = contract
            self.logger.info(f"Contract {name} deployed successfully with gas limit {gas_limit}")
            return True
        except Exception as e:
            self.logger.error(f"Contract deployment failed: {str(e)}")
            raise

    def _parse_contract(self, code):
        """Парсинг кода контракта"""
        try:
            # Базовые проверки синтаксиса
            if not isinstance(code, str):
                raise ValueError("Contract code must be a string")
            
            # Разделение на функции и переменные
            lines = code.strip().split('\n')
            contract = {
                'functions': {},
                'state': {},
                'events': []
            }
            
            current_function = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Обработка объявления функции
                if line.startswith('def '):
                    func_name = line.split('(')[0][4:].strip()
                    contract['functions'][func_name] = {
                        'code': line,
                        'params': self._parse_function_params(line)
                    }
                    current_function = func_name
                # Обработка состояния
                elif line.startswith('state.'):
                    var_name = line.split('=')[0].strip()[6:]
                    var_value = line.split('=')[1].strip()
                    contract['state'][var_name] = self._parse_value(var_value)
                # Обработка событий
                elif line.startswith('event '):
                    event_name = line[6:].strip()
                    contract['events'].append(event_name)
                    
            return contract
        except Exception as e:
            self.logger.error(f"Contract parsing failed: {str(e)}")
            raise

    def _parse_function_params(self, func_decl):
        """Парсинг параметров функции"""
        params_str = func_decl.split('(')[1].split(')')[0]
        return [p.strip() for p in params_str.split(',') if p.strip()]

    def _parse_value(self, value_str):
        """Парсинг значений переменных"""
        try:
            # Попытка преобразовать в число
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                # Если не число, возвращаем как строку
                return value_str.strip('"\'')

    def execute(self, contract_name, function_name, params=None, gas_limit=None):
        """Выполнение функции контракта с учетом газа"""
        try:
            if contract_name not in self.contracts:
                raise ValueError(f"Contract {contract_name} not found")
                
            contract = self.contracts[contract_name]
            if function_name not in contract['functions']:
                raise ValueError(f"Function {function_name} not found in contract {contract_name}")
                
            if gas_limit is None:
                gas_limit = contract.get('gas_limit', self.EXECUTION_LIMITS['MAX_GAS_PER_TX'])
                
            func = contract['functions'][function_name]
            
            # Проверка параметров
            if len(params or []) != len(func['params']):
                raise ValueError(f"Invalid number of parameters for function {function_name}")
                
            # Создание контекста выполнения
            context = {
                'state': contract['state'],
                'params': dict(zip(func['params'], params or [])),
                'events': [],
                'gas_used': 0,
                'gas_limit': gas_limit,
                'call_depth': 0
            }
            
            # Выполнение функции
            result = self._execute_function(func['code'], context)
            
            # Обновление состояния контракта
            self.contracts[contract_name]['state'] = context['state']
            
            # Логирование событий и использования газа
            for event in context['events']:
                self.logger.info(f"Event emitted in {contract_name}: {event}")
            self.logger.info(f"Gas used: {context['gas_used']}/{gas_limit}")
                
            return result
        except Exception as e:
            self.logger.error(f"Contract execution failed: {str(e)}")
            raise

    def _execute_function(self, func_code, context):
        """Выполнение кода функции с учетом газа"""
        try:
            # Проверка глубины вызова
            if context['call_depth'] >= self.EXECUTION_LIMITS['MAX_CALL_DEPTH']:
                raise ValueError("Maximum call depth exceeded")
                
            context['call_depth'] += 1
            
            # Создаем локальное пространство имен с функциями для учета газа
            local_vars = {
                'state': context['state'],
                'params': context['params'],
                'emit': lambda event: self._emit_event(context, event),
                'require': lambda condition, message: self._require(condition, message, context),
                'gas_left': lambda: context['gas_limit'] - context['gas_used']
            }
            
            # Выполняем код функции
            exec(func_code, {}, local_vars)
            
            # Возвращаем результат
            return local_vars.get('result')
        except Exception as e:
            self.logger.error(f"Function execution failed: {str(e)}")
            raise
        finally:
            context['call_depth'] -= 1

    def _use_gas(self, context, amount):
        """Использование газа"""
        if context['gas_used'] + amount > context['gas_limit']:
            raise ValueError("Out of gas")
        context['gas_used'] += amount

    def _emit_event(self, context, event):
        """Генерация события с учетом газа"""
        self._use_gas(context, self.GAS_COSTS['EVENT_EMIT'])
        context['events'].append(event)

    def _require(self, condition, message, context):
        """Проверка условия с учетом газа"""
        self._use_gas(context, self.GAS_COSTS['REQUIRE'])
        if not condition:
            raise ValueError(f"Requirement failed: {message}")

    def _read_state(self, context, key):
        """Чтение из состояния с учетом газа"""
        self._use_gas(context, self.GAS_COSTS['SLOAD'])
        return context['state'].get(key)

    def _write_state(self, context, key, value):
        """Запись в состояние с учетом газа"""
        self._use_gas(context, self.GAS_COSTS['SSTORE'])
        context['state'][key] = value

    def get_state(self, contract_name):
        """Получение состояния контракта"""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not found")
        return self.contracts[contract_name]['state']

    def get_events(self, contract_name):
        """Получение событий контракта"""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not found")
        return self.contracts[contract_name]['events']

# Пример использования DSL:
"""
contract SimpleToken
    require params.sender in state, "Sender not found"
    require params.amount > 0, "Amount must be positive"
    require state[params.sender] >= params.amount, "Insufficient funds"
    
    transfer params.sender, params.receiver, params.amount
    emit "Transfer", {"from": params.sender, "to": params.receiver, "amount": params.amount}
end
""" 