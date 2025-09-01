"""
Sistema de Monitoramento e Alertas para Gianna

Implementa métricas em tempo real, sistema de alertas, profiling automático
e dashboard de métricas para otimização de performance.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean, median, stdev
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Set, Union

import psutil
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import numpy as np

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/numpy não disponível - gráficos desabilitados")


class AlertLevel(Enum):
    """Níveis de alerta"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Tipos de métricas"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricData:
    """Dados de uma métrica"""

    name: str
    type: MetricType
    value: Union[float, int]
    timestamp: float
    labels: Dict[str, str] = None
    unit: str = ""

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class Alert:
    """Estrutura de alerta"""

    level: AlertLevel
    title: str
    message: str
    timestamp: float
    metric_name: str
    current_value: Any
    threshold: Any
    labels: Dict[str, str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class MetricCollector(ABC):
    """Interface para coletores de métricas"""

    @abstractmethod
    def collect(self) -> List[MetricData]:
        """Coleta métricas do sistema"""
        pass


class SystemMetricsCollector(MetricCollector):
    """Coletor de métricas do sistema"""

    def collect(self) -> List[MetricData]:
        metrics = []
        current_time = time.time()

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(
            MetricData(
                name="system.cpu.usage_percent",
                type=MetricType.GAUGE,
                value=cpu_percent,
                timestamp=current_time,
                unit="%",
            )
        )

        # Memória
        memory = psutil.virtual_memory()
        metrics.append(
            MetricData(
                name="system.memory.usage_percent",
                type=MetricType.GAUGE,
                value=memory.percent,
                timestamp=current_time,
                unit="%",
            )
        )

        metrics.append(
            MetricData(
                name="system.memory.available_mb",
                type=MetricType.GAUGE,
                value=memory.available / (1024 * 1024),
                timestamp=current_time,
                unit="MB",
            )
        )

        # Disco
        disk = psutil.disk_usage("/")
        metrics.append(
            MetricData(
                name="system.disk.usage_percent",
                type=MetricType.GAUGE,
                value=(disk.used / disk.total) * 100,
                timestamp=current_time,
                unit="%",
            )
        )

        # Rede
        net_io = psutil.net_io_counters()
        metrics.append(
            MetricData(
                name="system.network.bytes_sent",
                type=MetricType.COUNTER,
                value=net_io.bytes_sent,
                timestamp=current_time,
                unit="bytes",
            )
        )

        metrics.append(
            MetricData(
                name="system.network.bytes_recv",
                type=MetricType.COUNTER,
                value=net_io.bytes_recv,
                timestamp=current_time,
                unit="bytes",
            )
        )

        return metrics


class ApplicationMetricsCollector(MetricCollector):
    """Coletor de métricas da aplicação"""

    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self.lock = Lock()

    def increment_counter(
        self, name: str, value: int = 1, labels: Dict[str, str] = None
    ):
        """Incrementa contador"""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Define valor de gauge"""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Registra valor em histograma"""
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            # Mantém apenas últimos 1000 valores
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Registra tempo de execução"""
        with self.lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
            # Mantém apenas últimos 1000 valores
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Cria chave única para métrica"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _parse_key(self, key: str) -> tuple:
        """Parseia chave em nome e labels"""
        if "{" not in key:
            return key, {}

        name, label_part = key.split("{", 1)
        label_part = label_part.rstrip("}")

        labels = {}
        if label_part:
            for pair in label_part.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v

        return name, labels

    def collect(self) -> List[MetricData]:
        metrics = []
        current_time = time.time()

        with self.lock:
            # Counters
            for key, value in self.counters.items():
                name, labels = self._parse_key(key)
                metrics.append(
                    MetricData(
                        name=name,
                        type=MetricType.COUNTER,
                        value=value,
                        timestamp=current_time,
                        labels=labels,
                    )
                )

            # Gauges
            for key, value in self.gauges.items():
                name, labels = self._parse_key(key)
                metrics.append(
                    MetricData(
                        name=name,
                        type=MetricType.GAUGE,
                        value=value,
                        timestamp=current_time,
                        labels=labels,
                    )
                )

            # Histograms (estatísticas)
            for key, values in self.histograms.items():
                if values:
                    name, labels = self._parse_key(key)

                    # Estatísticas básicas
                    metrics.extend(
                        [
                            MetricData(
                                f"{name}.count",
                                MetricType.GAUGE,
                                len(values),
                                current_time,
                                labels,
                            ),
                            MetricData(
                                f"{name}.mean",
                                MetricType.GAUGE,
                                mean(values),
                                current_time,
                                labels,
                            ),
                            MetricData(
                                f"{name}.median",
                                MetricType.GAUGE,
                                median(values),
                                current_time,
                                labels,
                            ),
                            MetricData(
                                f"{name}.min",
                                MetricType.GAUGE,
                                min(values),
                                current_time,
                                labels,
                            ),
                            MetricData(
                                f"{name}.max",
                                MetricType.GAUGE,
                                max(values),
                                current_time,
                                labels,
                            ),
                        ]
                    )

                    if len(values) > 1:
                        metrics.append(
                            MetricData(
                                f"{name}.stddev",
                                MetricType.GAUGE,
                                stdev(values),
                                current_time,
                                labels,
                            )
                        )

            # Timers (similar aos histogramas)
            for key, durations in self.timers.items():
                if durations:
                    name, labels = self._parse_key(key)

                    metrics.extend(
                        [
                            MetricData(
                                f"{name}.count",
                                MetricType.GAUGE,
                                len(durations),
                                current_time,
                                labels,
                                "requests",
                            ),
                            MetricData(
                                f"{name}.mean",
                                MetricType.GAUGE,
                                mean(durations),
                                current_time,
                                labels,
                                "ms",
                            ),
                            MetricData(
                                f"{name}.median",
                                MetricType.GAUGE,
                                median(durations),
                                current_time,
                                labels,
                                "ms",
                            ),
                            MetricData(
                                f"{name}.p95",
                                MetricType.GAUGE,
                                self._percentile(durations, 95),
                                current_time,
                                labels,
                                "ms",
                            ),
                            MetricData(
                                f"{name}.p99",
                                MetricType.GAUGE,
                                self._percentile(durations, 99),
                                current_time,
                                labels,
                                "ms",
                            ),
                        ]
                    )

        return metrics

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calcula percentil"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c


class Profiler:
    """Profiler automático para operações lentas"""

    def __init__(
        self,
        slow_threshold: float = 1.0,
        sample_rate: float = 0.1,
        max_profiles: int = 100,
    ):
        self.slow_threshold = slow_threshold
        self.sample_rate = sample_rate
        self.max_profiles = max_profiles
        self.profiles = deque(maxlen=max_profiles)
        self.lock = Lock()

    def profile_function(self, func: Callable, *args, **kwargs):
        """Perfila execução de função"""
        import cProfile
        import io
        import pstats

        should_profile = time.time() % (1 / self.sample_rate) < 1

        start_time = time.time()

        if should_profile:
            pr = cProfile.Profile()
            pr.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            duration = end_time - start_time

            if should_profile:
                pr.disable()

                if duration > self.slow_threshold:
                    # Captura profile
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s)
                    ps.sort_stats("cumulative")
                    ps.print_stats(20)  # Top 20 funções

                    profile_data = {
                        "function": func.__name__,
                        "duration": duration,
                        "timestamp": end_time,
                        "profile": s.getvalue(),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs),
                    }

                    with self.lock:
                        self.profiles.append(profile_data)

                    logger.warning(
                        f"Slow operation detected: {func.__name__} ({duration:.2f}s)"
                    )

        return result

    def get_slow_operations(self, limit: int = 10) -> List[Dict]:
        """Obtém operações mais lentas"""
        with self.lock:
            sorted_profiles = sorted(
                self.profiles, key=lambda x: x["duration"], reverse=True
            )
            return list(sorted_profiles[:limit])


class AlertManager:
    """Gerenciador de alertas"""

    def __init__(self):
        self.rules = []
        self.alerts = deque(maxlen=1000)
        self.handlers = defaultdict(list)
        self.lock = Lock()

    def add_rule(
        self,
        metric_name: str,
        condition: str,
        threshold: Union[float, int],
        level: AlertLevel,
        title: str,
        message: str,
    ):
        """
        Adiciona regra de alerta

        Args:
            metric_name: Nome da métrica
            condition: '>', '<', '>=', '<=', '==', '!='
            threshold: Valor limite
            level: Nível de alerta
            title: Título do alerta
            message: Mensagem do alerta
        """
        rule = {
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "level": level,
            "title": title,
            "message": message,
            "last_triggered": 0,
            "cooldown": 300,  # 5 minutos
        }
        self.rules.append(rule)

    def add_handler(self, level: AlertLevel, handler: Callable[[Alert], None]):
        """Adiciona handler para nível de alerta"""
        self.handlers[level].append(handler)

    def check_metrics(self, metrics: List[MetricData]):
        """Verifica métricas contra regras de alerta"""
        current_time = time.time()

        for metric in metrics:
            for rule in self.rules:
                if self._matches_rule(metric, rule, current_time):
                    alert = Alert(
                        level=rule["level"],
                        title=rule["title"],
                        message=rule["message"].format(
                            metric_name=metric.name,
                            current_value=metric.value,
                            threshold=rule["threshold"],
                        ),
                        timestamp=current_time,
                        metric_name=metric.name,
                        current_value=metric.value,
                        threshold=rule["threshold"],
                        labels=metric.labels,
                    )

                    self._trigger_alert(alert, rule)

    def _matches_rule(
        self, metric: MetricData, rule: Dict, current_time: float
    ) -> bool:
        """Verifica se métrica dispara regra"""
        if metric.name != rule["metric_name"]:
            return False

        # Cooldown
        if current_time - rule["last_triggered"] < rule["cooldown"]:
            return False

        # Condição
        condition = rule["condition"]
        threshold = rule["threshold"]
        value = metric.value

        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold

        return False

    def _trigger_alert(self, alert: Alert, rule: Dict):
        """Dispara alerta"""
        with self.lock:
            self.alerts.append(alert)
            rule["last_triggered"] = alert.timestamp

        # Chama handlers
        for handler in self.handlers[alert.level]:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Erro em handler de alerta: {e}")

        logger.log(
            "ERROR" if alert.level == AlertLevel.CRITICAL else "WARNING",
            f"Alert [{alert.level.value}]: {alert.title} - {alert.message}",
        )

    def get_active_alerts(self, minutes: int = 60) -> List[Alert]:
        """Obtém alertas ativos dos últimos N minutos"""
        cutoff_time = time.time() - (minutes * 60)

        with self.lock:
            return [alert for alert in self.alerts if alert.timestamp > cutoff_time]


class PerformanceDashboard:
    """Dashboard de métricas de performance"""

    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.max_history = 1000
        self.lock = Lock()

    def record_metrics(self, metrics: List[MetricData]):
        """Registra métricas no histórico"""
        with self.lock:
            for metric in metrics:
                key = f"{metric.name}_{json.dumps(metric.labels, sort_keys=True)}"
                self.metrics_history[key].append(
                    {"timestamp": metric.timestamp, "value": metric.value}
                )

                # Mantém histórico limitado
                if len(self.metrics_history[key]) > self.max_history:
                    self.metrics_history[key].popleft()

    def get_metric_summary(self, metric_name: str, minutes: int = 60) -> Dict[str, Any]:
        """Obtém resumo de uma métrica"""
        cutoff_time = time.time() - (minutes * 60)

        with self.lock:
            relevant_keys = [
                k for k in self.metrics_history.keys() if k.startswith(metric_name)
            ]

            if not relevant_keys:
                return {"error": f"Metric {metric_name} not found"}

            all_values = []
            for key in relevant_keys:
                values = [
                    point["value"]
                    for point in self.metrics_history[key]
                    if point["timestamp"] > cutoff_time
                ]
                all_values.extend(values)

            if not all_values:
                return {"error": "No data in time range"}

            return {
                "metric_name": metric_name,
                "time_range_minutes": minutes,
                "count": len(all_values),
                "mean": mean(all_values),
                "median": median(all_values),
                "min": min(all_values),
                "max": max(all_values),
                "stddev": stdev(all_values) if len(all_values) > 1 else 0,
            }

    def generate_report(self, minutes: int = 60) -> Dict[str, Any]:
        """Gera relatório completo"""
        important_metrics = [
            "system.cpu.usage_percent",
            "system.memory.usage_percent",
            "application.llm.response_time",
            "application.cache.hit_rate",
        ]

        report = {
            "timestamp": time.time(),
            "time_range_minutes": minutes,
            "metrics": {},
        }

        for metric in important_metrics:
            summary = self.get_metric_summary(metric, minutes)
            if "error" not in summary:
                report["metrics"][metric] = summary

        return report

    def plot_metric(self, metric_name: str, minutes: int = 60, save_path: str = None):
        """Gera gráfico de métrica"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting não disponível")
            return None

        cutoff_time = time.time() - (minutes * 60)

        with self.lock:
            relevant_keys = [
                k for k in self.metrics_history.keys() if k.startswith(metric_name)
            ]

            if not relevant_keys:
                logger.warning(f"Métrica {metric_name} não encontrada")
                return None

            plt.figure(figsize=(12, 6))

            for key in relevant_keys:
                points = [
                    point
                    for point in self.metrics_history[key]
                    if point["timestamp"] > cutoff_time
                ]

                if points:
                    timestamps = [p["timestamp"] for p in points]
                    values = [p["value"] for p in points]

                    # Converte timestamp para tempo relativo
                    start_time = min(timestamps)
                    relative_times = [
                        (t - start_time) / 60 for t in timestamps
                    ]  # minutos

                    plt.plot(
                        relative_times, values, label=key, marker="o", markersize=2
                    )

            plt.title(f"Metric: {metric_name} (Last {minutes} minutes)")
            plt.xlabel("Time (minutes ago)")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

            plt.close()


class PerformanceMonitor:
    """Monitor principal de performance"""

    def __init__(self, collection_interval: int = 30, enable_profiling: bool = True):

        self.collection_interval = collection_interval
        self.enable_profiling = enable_profiling

        # Componentes
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard = PerformanceDashboard()

        if enable_profiling:
            self.profiler = Profiler()
        else:
            self.profiler = None

        # Controle
        self.active = False
        self.monitoring_thread = None

        # Setup inicial
        self._setup_default_alerts()
        self._setup_default_handlers()

    def _setup_default_alerts(self):
        """Configura alertas padrão"""
        # CPU
        self.alert_manager.add_rule(
            "system.cpu.usage_percent",
            ">",
            80,
            AlertLevel.WARNING,
            "High CPU Usage",
            "CPU usage is {current_value}%, threshold: {threshold}%",
        )

        self.alert_manager.add_rule(
            "system.cpu.usage_percent",
            ">",
            95,
            AlertLevel.CRITICAL,
            "Critical CPU Usage",
            "CPU usage is critically high: {current_value}%",
        )

        # Memória
        self.alert_manager.add_rule(
            "system.memory.usage_percent",
            ">",
            85,
            AlertLevel.WARNING,
            "High Memory Usage",
            "Memory usage is {current_value}%, threshold: {threshold}%",
        )

        self.alert_manager.add_rule(
            "system.memory.usage_percent",
            ">",
            95,
            AlertLevel.CRITICAL,
            "Critical Memory Usage",
            "Memory usage is critically high: {current_value}%",
        )

    def _setup_default_handlers(self):
        """Configura handlers padrão de alertas"""

        def log_handler(alert: Alert):
            logger.log(
                (
                    "ERROR"
                    if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
                    else "WARNING"
                ),
                f"[{alert.level.value.upper()}] {alert.title}: {alert.message}",
            )

        # Adiciona handler para todos os níveis
        for level in AlertLevel:
            self.alert_manager.add_handler(level, log_handler)

    def start_monitoring(self):
        """Inicia monitoramento"""
        if self.active:
            return

        self.active = True
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Para monitoramento"""
        self.active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.active:
            try:
                # Coleta métricas
                metrics = []
                metrics.extend(self.system_collector.collect())
                metrics.extend(self.app_collector.collect())

                # Verifica alertas
                self.alert_manager.check_metrics(metrics)

                # Registra no dashboard
                self.dashboard.record_metrics(metrics)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)

    def get_status(self) -> Dict[str, Any]:
        """Obtém status completo do monitoramento"""
        return {
            "monitoring_active": self.active,
            "collection_interval": self.collection_interval,
            "profiling_enabled": self.enable_profiling,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "metrics_collected": len(self.dashboard.metrics_history),
            "slow_operations": (
                len(self.profiler.get_slow_operations()) if self.profiler else 0
            ),
        }

    def profile_function(self, func: Callable, *args, **kwargs):
        """Perfila função se profiling habilitado"""
        if self.profiler:
            return self.profiler.profile_function(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    # Métodos de conveniência para métricas da aplicação
    def increment_counter(
        self, name: str, value: int = 1, labels: Dict[str, str] = None
    ):
        """Incrementa contador"""
        self.app_collector.increment_counter(name, value, labels)

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Define gauge"""
        self.app_collector.set_gauge(name, value, labels)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Registra histograma"""
        self.app_collector.record_histogram(name, value, labels)

    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Registra timer"""
        self.app_collector.record_timer(name, duration, labels)

    # Context manager para timing
    def timer_context(self, name: str, labels: Dict[str, str] = None):
        """Context manager para timing"""
        return TimerContext(self, name, labels)


class TimerContext:
    """Context manager para medir tempo de execução"""

    def __init__(
        self, monitor: PerformanceMonitor, name: str, labels: Dict[str, str] = None
    ):
        self.monitor = monitor
        self.name = name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (time.time() - self.start_time) * 1000  # em ms
            self.monitor.record_timer(self.name, duration, self.labels)
