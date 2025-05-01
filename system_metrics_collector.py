import psutil
import time
from datetime import datetime
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    def __init__(self):
        self.metrics_file = 'system_metrics.log'
        self.metrics = []
        self.max_entries = 1000  # Максимальное количество записей в логе

    def collect_metrics(self):
        """Сбор системных метрик"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                },
                'network': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv,
                    'packets_sent': psutil.net_io_counters().packets_sent,
                    'packets_recv': psutil.net_io_counters().packets_recv
                }
            }
            
            self.metrics.append(metrics)
            
            # Ограничиваем количество записей
            if len(self.metrics) > self.max_entries:
                self.metrics = self.metrics[-self.max_entries:]
            
            # Сохраняем метрики в файл
            self.save_metrics()
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return None

    def save_metrics(self):
        """Сохранение метрик в файл"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving system metrics: {str(e)}")

    def load_metrics(self):
        """Загрузка метрик из файла"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading system metrics: {str(e)}")

    def get_metrics_report(self, start_time=None, end_time=None):
        """Получение отчета о метриках за указанный период"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(hours=1)
            if not end_time:
                end_time = datetime.now()

            filtered_metrics = [
                m for m in self.metrics
                if start_time <= datetime.fromisoformat(m['timestamp']) <= end_time
            ]

            if not filtered_metrics:
                return {
                    'status': 'success',
                    'metrics': {
                        'cpu': {'average': 0, 'max': 0},
                        'memory': {'average': 0, 'max': 0},
                        'disk': {'average': 0, 'max': 0},
                        'network': {'average_sent': 0, 'average_recv': 0}
                    },
                    'history': {
                        'timestamps': [],
                        'cpu': [],
                        'memory': [],
                        'disk': [],
                        'network_sent': [],
                        'network_recv': []
                    }
                }

            # Вычисляем средние и максимальные значения
            cpu_values = [m['cpu']['percent'] for m in filtered_metrics]
            memory_values = [m['memory']['percent'] for m in filtered_metrics]
            disk_values = [m['disk']['percent'] for m in filtered_metrics]
            network_sent = [m['network']['bytes_sent'] for m in filtered_metrics]
            network_recv = [m['network']['bytes_recv'] for m in filtered_metrics]

            report = {
                'status': 'success',
                'metrics': {
                    'cpu': {
                        'average': sum(cpu_values) / len(cpu_values),
                        'max': max(cpu_values)
                    },
                    'memory': {
                        'average': sum(memory_values) / len(memory_values),
                        'max': max(memory_values)
                    },
                    'disk': {
                        'average': sum(disk_values) / len(disk_values),
                        'max': max(disk_values)
                    },
                    'network': {
                        'average_sent': sum(network_sent) / len(network_sent),
                        'average_recv': sum(network_recv) / len(network_recv)
                    }
                },
                'history': {
                    'timestamps': [m['timestamp'] for m in filtered_metrics],
                    'cpu': cpu_values,
                    'memory': memory_values,
                    'disk': disk_values,
                    'network_sent': network_sent,
                    'network_recv': network_recv
                }
            }

            return report
        except Exception as e:
            logger.error(f"Error generating system metrics report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            } 