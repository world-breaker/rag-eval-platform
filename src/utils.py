"""工具模块：配置加载、日志、计时"""

import yaml
import logging
import time
from pathlib import Path
from functools import wraps


SUPPORTED_VECTOR_STORES = ("numpy",)
SUPPORTED_RERANK_TYPES = ("cross_encoder",)


def validate_config(config: dict) -> dict:
    """校验配置里「写了但代码不支持」的选项，遇到就明确失败。

    为什么不在运行时静默忽略：`retrieval.vector_store: chromadb` 这类预留项
    永远不会生效——配置里写着它、实际跑的却是内置 NumPy 检索，属于**配置不诚实**。
    宁可启动即报错并给出替代方案，也不要让人以为向量库换了。

    同样的道理适用于 `retrieval.hybrid.rerank.type`：当前只实现了 Cross-Encoder 重排，
    写别的类型会**静默按 cross_encoder 跑**（等于骗人），所以这里 fail-fast。
    """
    retrieval = (config or {}).get("retrieval") or {}
    store = retrieval.get("vector_store", "numpy")
    if store not in SUPPORTED_VECTOR_STORES:
        raise ValueError(
            f"retrieval.vector_store={store!r} 未接入代码（当前只有 {SUPPORTED_VECTOR_STORES[0]}）。"
            "chromadb/Faiss 都只是扩展计划，未实现；请把 config.yaml 的 "
            "retrieval.vector_store 改回 'numpy'。"
        )

    rerank_cfg = ((retrieval.get("hybrid") or {}).get("rerank") or {})
    rerank_type = rerank_cfg.get("type", SUPPORTED_RERANK_TYPES[0])
    if rerank_type not in SUPPORTED_RERANK_TYPES:
        raise ValueError(
            f"retrieval.hybrid.rerank.type={rerank_type!r} 未接入代码"
            f"（当前只有 {SUPPORTED_RERANK_TYPES[0]}）。"
            "请把 config.yaml 的 retrieval.hybrid.rerank.type 改回 'cross_encoder'。"
        )
    return config


def load_config(path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件（加载即校验，不支持的选项立即失败）"""
    config_path = Path(path)
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))


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
