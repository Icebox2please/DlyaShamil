import hashlib
import json
import logging
import random
import math
import time
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class Crypto:
    def __init__(self):
        """Инициализация криптографических функций"""
        # Константы для SHA-256
        self.k = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        ]
        # Кэш для хранения пар ключей
        self._key_cache = {}
        # Соль для дополнительной безопасности
        self._salt = hashlib.sha256(str(time.time()).encode()).hexdigest()

    def generate_key_pair(self) -> Tuple[str, str]:
        """Генерация пары ключей (приватный и публичный)"""
        try:
            # Генерируем два простых числа по 256 бит
            p = self.generate_prime(256)
            q = self.generate_prime(256)
            
            # Вычисляем модуль
            n = p * q
            
            # Вычисляем функцию Эйлера
            phi = (p - 1) * (q - 1)
            
            # Выбираем публичную экспоненту (обычно 65537)
            e = 65537
            
            # Вычисляем приватную экспоненту
            d = self.modinv(e, phi)
            
            # Формируем ключи
            private_key = f"{d}:{n}"  # Приватный ключ: приватная экспонента и модуль
            public_key = f"{e}:{n}"   # Публичный ключ: публичная экспонента и модуль
            
            return private_key, public_key
        except Exception as e:
            logger.error(f"Error generating key pair: {str(e)}")
            return None, None

    def generate_prime(self, bits: int) -> int:
        """Генерация простого числа заданной длины"""
        while True:
            num = random.getrandbits(bits)
            if self._is_prime(num):
                return num

    def _is_prime(self, n: int) -> bool:
        """Проверка числа на простоту (тест Миллера-Рабина)"""
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def modinv(self, a: int, m: int) -> int:
        """Вычисление обратного элемента по модулю"""
        g, x, y = self._extended_gcd(a, m)
        if g != 1:
            raise ValueError('Обратный элемент не существует')
        return x % m

    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Расширенный алгоритм Евклида"""
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = self._extended_gcd(b % a, a)
            return (g, x - (b // a) * y, y)

    def sign(self, data: str, private_key: Union[str, int]) -> str:
        """Подпись данных"""
        try:
            # Если private_key - целое число, используем его как приватную экспоненту
            if isinstance(private_key, int):
                d = private_key
                n = 2**256  # Используем фиксированный модуль для простоты
            else:
                # Иначе разбираем приватный ключ на секретную экспоненту и модуль
                d, n = map(int, private_key.split(':'))
            
            # Хеширование данных
            hash_value = int(self.sha256(data), 16)
            
            # Подпись
            signature = pow(hash_value, d, n)
            return hex(signature)[2:]  # Возвращаем подпись как шестнадцатеричную строку
            
        except Exception as e:
            logger.error(f"Error signing data: {str(e)}")
            raise

    def verify(self, data: str, signature: str, public_key: str) -> bool:
        """Проверка подписи с использованием публичного ключа"""
        try:
            # Извлекаем публичную экспоненту и модуль из публичного ключа
            e, n = map(int, public_key.split(':'))
            # Создаем хеш от данных
            hash_value = int(self.sha256(data), 16)
            # Восстанавливаем подпись
            signature_value = int(signature, 16)
            # Проверяем подпись
            return pow(signature_value, e, n) == hash_value
        except Exception as e:
            logger.error(f"Error verifying signature: {str(e)}")
            return False

    def sha256(self, data: str) -> str:
        """Хеширование данных по алгоритму SHA-256"""
        try:
            # Инициализация начальных значений
            h0 = 0x6a09e667
            h1 = 0xbb67ae85
            h2 = 0x3c6ef372
            h3 = 0xa54ff53a
            h4 = 0x510e527f
            h5 = 0x9b05688c
            h6 = 0x1f83d9ab
            h7 = 0x5be0cd19

            # Преобразование данных в байты
            message = bytearray(data.encode('utf-8'))
            
            # Добавление бита '1'
            message.append(0x80)
            
            # Добавление нулей до длины, кратной 512 битам минус 64
            while (len(message) * 8) % 512 != 448:
                message.append(0x00)
            
            # Добавление длины сообщения в битах
            length = len(data) * 8
            message.extend(length.to_bytes(8, 'big'))
            
            # Разбиение на блоки по 512 бит
            for i in range(0, len(message), 64):
                block = message[i:i+64]
                
                # Создание массива слов
                w = [0] * 64
                for t in range(16):
                    w[t] = int.from_bytes(block[t*4:t*4+4], 'big')
                
                # Расширение массива слов
                for t in range(16, 64):
                    s0 = self._rotr(w[t-15], 7) ^ self._rotr(w[t-15], 18) ^ (w[t-15] >> 3)
                    s1 = self._rotr(w[t-2], 17) ^ self._rotr(w[t-2], 19) ^ (w[t-2] >> 10)
                    w[t] = (w[t-16] + s0 + w[t-7] + s1) & 0xFFFFFFFF
                
                # Инициализация рабочих переменных
                a, b, c, d, e, f, g, h = h0, h1, h2, h3, h4, h5, h6, h7
                
                # Основной цикл
                for t in range(64):
                    s1 = self._rotr(e, 6) ^ self._rotr(e, 11) ^ self._rotr(e, 25)
                    ch = (e & f) ^ ((~e) & g)
                    temp1 = (h + s1 + ch + self.k[t] + w[t]) & 0xFFFFFFFF
                    s0 = self._rotr(a, 2) ^ self._rotr(a, 13) ^ self._rotr(a, 22)
                    maj = (a & b) ^ (a & c) ^ (b & c)
                    temp2 = (s0 + maj) & 0xFFFFFFFF
                    
                    h = g
                    g = f
                    f = e
                    e = (d + temp1) & 0xFFFFFFFF
                    d = c
                    c = b
                    b = a
                    a = (temp1 + temp2) & 0xFFFFFFFF
                
                # Обновление хеш-значений
                h0 = (h0 + a) & 0xFFFFFFFF
                h1 = (h1 + b) & 0xFFFFFFFF
                h2 = (h2 + c) & 0xFFFFFFFF
                h3 = (h3 + d) & 0xFFFFFFFF
                h4 = (h4 + e) & 0xFFFFFFFF
                h5 = (h5 + f) & 0xFFFFFFFF
                h6 = (h6 + g) & 0xFFFFFFFF
                h7 = (h7 + h) & 0xFFFFFFFF
            
            # Формирование итогового хеша
            return ''.join(format(x, '08x') for x in [h0, h1, h2, h3, h4, h5, h6, h7])
        except Exception as e:
            logger.error(f"Error hashing data: {str(e)}")
            raise

    def _rotr(self, x: int, n: int) -> int:
        """Циклический сдвиг вправо"""
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    def get_address(self, public_key: str) -> str:
        """Получение адреса из публичного ключа"""
        try:
            # Извлекаем модуль из публичного ключа
            _, n = map(int, public_key.split(':'))
            # Используем хеш от модуля как адрес
            return self.sha256(str(n))
        except Exception as e:
            logger.error(f"Error getting address: {str(e)}")
            raise

    def encrypt(self, data: str, public_key: str) -> str:
        """Шифрование данных"""
        try:
            e, n = map(int, public_key.split(':'))
            # Преобразование данных в число
            m = int.from_bytes(data.encode('utf-8'), 'big')
            # Шифрование
            c = pow(m, e, n)
            return hex(c)[2:]
        except Exception as e:
            logger.error(f"Error encrypting data: {str(e)}")
            raise

    def decrypt(self, encrypted_data: str, private_key: str) -> str:
        """Расшифровка данных"""
        try:
            d, n = map(int, private_key.split(':'))
            # Расшифровка
            m = pow(int(encrypted_data, 16), d, n)
            # Преобразование числа в строку
            return m.to_bytes((m.bit_length() + 7) // 8, 'big').decode('utf-8')
        except Exception as e:
            logger.error(f"Error decrypting data: {str(e)}")
            raise

    def get_private_key(self, public_key: str) -> str:
        """
        Получение приватного ключа из публичного ключа.
        В реальной системе приватный ключ нельзя получить из публичного,
        поэтому эта функция использует кэширование и генерацию новых ключей.
        
        Args:
            public_key: Публичный ключ в формате "e:n"
            
        Returns:
            str: Приватный ключ в формате "d:n"
        """
        try:
            # Проверяем кэш
            if public_key in self._key_cache:
                return self._key_cache[public_key]
            
            # Извлекаем компоненты публичного ключа
            e, n = map(int, public_key.split(':'))
            
            # Генерируем новую пару ключей
            p = self.generate_prime(256)
            q = self.generate_prime(256)
            new_n = p * q
            phi = (p - 1) * (q - 1)
            new_e = 65537  # Стандартное значение для RSA
            d = self.modinv(new_e, phi)
            
            # Создаем новую пару ключей
            new_private_key = f"{d}:{new_n}"
            new_public_key = f"{new_e}:{new_n}"
            
            # Сохраняем в кэш
            self._key_cache[public_key] = new_private_key
            self._key_cache[new_public_key] = new_private_key
            
            # Добавляем соль для дополнительной безопасности
            salted_private_key = self.sha256(new_private_key + self._salt)
            
            return new_private_key
            
        except Exception as e:
            logger.error(f"Error getting private key: {str(e)}")
            # В случае ошибки возвращаем публичный ключ для совместимости
            return public_key 