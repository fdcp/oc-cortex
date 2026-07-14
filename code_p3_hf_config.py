"""
HuggingFace 模型加载配置
设置离线模式和缓存目录，必须在导入 sentence_transformers/transformers 之前调用
"""
import os
from typing import Optional


def setup_hf_env(offline_mode: bool = True, cache_folder: Optional[str] = None):
    """
    设置 HuggingFace 环境变量
    必须在导入 sentence_transformers 或 transformers 之前调用

    Args:
        offline_mode: True 表示离线模式，仅使用本地缓存
        cache_folder: 模型缓存目录，None 使用默认值 ~/.cache/huggingface/hub
    """
    # 离线模式
    if offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # 缓存目录
    if cache_folder:
        os.environ["HF_HOME"] = cache_folder
        os.environ["TRANSFORMERS_CACHE"] = cache_folder
        os.environ["HF_HUB_CACHE"] = cache_folder
    else:
        default_cache = os.path.expanduser("~/.cache/huggingface/hub")
        os.environ["HF_HOME"] = default_cache
        os.environ["TRANSFORMERS_CACHE"] = default_cache
        os.environ["HF_HUB_CACHE"] = default_cache
