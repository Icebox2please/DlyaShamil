import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import os

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Класс для сбора и хранения метрик производительности"""
    
    def __init__(self):
        self.metrics_file = 'performance_metrics.json'
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Загрузка метрик из файла"""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {str(e)}")
        return {
            'blockchain_metrics': [],
            'node_metrics': [],
            'api_metrics': []
        }
    
    def _save_metrics(self):
        """Сохранение метрик в файл"""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def collect_blockchain_metrics(self, metrics: Dict[str, Any]):
        """Сбор метрик блокчейна"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics['blockchain_metrics'].append(metrics)
        self._save_metrics()
    
    def collect_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Сбор метрик узла"""
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['node_id'] = node_id
        self.metrics['node_metrics'].append(metrics)
        self._save_metrics()
    
    def collect_api_metrics(self, endpoint: str, metrics: Dict[str, Any]):
        """Сбор метрик API"""
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['endpoint'] = endpoint
        self.metrics['api_metrics'].append(metrics)
        self._save_metrics()
    
    def generate_performance_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Генерация отчета о производительности"""
        report = {
            'blockchain_metrics': self._filter_metrics('blockchain_metrics', start_time, end_time),
            'node_metrics': self._filter_metrics('node_metrics', start_time, end_time),
            'api_metrics': self._filter_metrics('api_metrics', start_time, end_time)
        }
        
        # Добавляем агрегированные метрики
        report['summary'] = {
            'total_blocks': len(report['blockchain_metrics']),
            'total_transactions': sum(m.get('transactions_count', 0) for m in report['blockchain_metrics']),
            'average_block_time': self._calculate_average(report['blockchain_metrics'], 'block_time'),
            'average_api_response_time': self._calculate_average(report['api_metrics'], 'response_time')
        }
        
        return report
    
    def _filter_metrics(self, metric_type: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Фильтрация метрик по временному диапазону"""
        return [
            m for m in self.metrics[metric_type]
            if start_time <= datetime.fromisoformat(m['timestamp']) <= end_time
        ]
    
    def _calculate_average(self, metrics: List[Dict[str, Any]], field: str) -> float:
        """Расчет среднего значения для поля"""
        values = [m.get(field, 0) for m in metrics if field in m]
        return sum(values) / len(values) if values else 0 