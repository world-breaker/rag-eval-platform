"""工具模块：配置加载、日志、计时"""

import yaml
import logging
import time
from pathlib import Path
from functools import wraps


def load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件"""
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """统一日志配置"""
    logger = logging.getLogger("rag_eval")
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger


logger = setup_logging()


def timer(func):
    """装饰器：计时并输出日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} 耗时 {elapsed:.3f}s")
        return result

    return wrapper
