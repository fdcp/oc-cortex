"""
Phase 5 核心模块: 知识图谱构建
- 三元组抽取 (LLM): 从 task summary 中抽取 (head, relation, tail) 三元组
- 实体对齐 (Qdrant + LLM): embedding 相似度 + LLM 二次确认
- 图谱构建 (NetworkX): MultiDiGraph + gpickle 持久化
"""
import json
import os
import re
import time
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from typing import List, Tuple

import yaml
from openai import OpenAI
from loguru import logger
import networkx as nx
import numpy as np

from code_update_prompt_utils import load_prompt

from code_p2_models import Task
from code_p5_models import Triple, Entity, KGStats

if TYPE_CHECKING:
    from code_p1_utils import Config


# 默认 Zen 端点 (opencode_models.yaml 加载失败时回退)
DEFAULT_FALLBACK_BASE_URL = "https://opencode.ai/zen/v1"
DEFAULT_FALLBACK_API_KEY_ENV = "OPENCODE_ZEN_API_KEY"


# ============================================================
# 线程级速率限制器
# ============================================================

class _LLMRateLimiter:
    """线程级速率限制器,所有 LLM 调用共享一个令牌桶。

    简单实现: 每次 acquire() 时, 若距上次调用 < min_interval, sleep 补齐。
    N 个并发线程会自然错开, 不会再瞬间打爆 rate limit。

    典型用法:
        limiter = _LLMRateLimiter(rate_per_sec=2.0)  # 至少 0.5s 间隔
        for _ in range(N):
            t = threading.Thread(target=worker, args=(limiter,))
            t.start()
    """

    def __init__(self, rate_per_sec: float):
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec 必须 > 0, 当前: {rate_per_sec}")
        self.min_interval = 1.0 / rate_per_sec
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def acquire(self) -> None:
        """获取令牌: 若距上次调用不足 min_interval, sleep 补齐"""
        with self._lock:
            now = time.time()
            wait = self._last_call_ts + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.time()


# ============================================================
# 辅助函数
# ============================================================

def _stable_uuid(text: str) -> str:
    """从字符串生成稳定的 UUID (基于 MD5)"""
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def _extract_json_from_response(response: str) -> str:
    """从 LLM 响应中提取 JSON 字符串

    策略:
    1. 直接解析 (如果以 { 或 [ 开头)
    2. markdown 代码块提取
    3. json.JSONDecoder().raw_decode() 提取第一个合法 JSON 对象/数组
    4. 回退: 第一个 { / [ 到最后一个 } / ]
    """
    response = response.strip()

    # 1. 直接尝试 (支持 object 和 array)
    if response.startswith(("{", "[")):
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # 可能有尾部多余内容, 用 raw_decode 提取
            pass

    # 2. markdown 代码块
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            pass

    # 3. raw_decode: 优先从 [ 开始 (数组), 其次从 { 开始 (对象)
    first_obj = response.find("{")
    first_arr = response.find("[")
    decoder = json.JSONDecoder()

    # 尝试所有可能的起始位置, 取最长有效 JSON
    positions = sorted(set(p for p in [first_arr, first_obj] if p != -1))
    best_result = None
    best_len = 0

    for pos in positions:
        try:
            obj, end_idx = decoder.raw_decode(response, pos)
            json_text = json.dumps(obj, ensure_ascii=False)
            if len(json_text) > best_len:
                best_result = json_text
                best_len = len(json_text)
        except json.JSONDecodeError:
            pass

    if best_result:
        return best_result

    # 4. 回退: 第一个 { 到最后一个 }, 或第一个 [ 到最后一个 ]
    brace_start = response.find("{")
    brace_end = response.rfind("}")
    bracket_start = response.find("[")
    bracket_end = response.rfind("]")

    # 优先选择更长的匹配
    options = []
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        options.append((brace_start, brace_end + 1))
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        options.append((bracket_start, bracket_end + 1))

    if options:
        best = max(options, key=lambda x: x[1] - x[0])
        return response[best[0]:best[1]]

    return response


def _repair_truncated_json(json_str: str) -> str:
    """修复被截断的 JSON"""
    stack = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        c = json_str[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape_next = True
            i += 1
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if c in ('{', '['):
                stack.append(c)
            elif c == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif c == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
        i += 1

    if not stack and not in_string:
        return json_str

    result = json_str.rstrip()
    if in_string:
        result += '"'

    last_comma = result.rfind(',')
    if last_comma > 0:
        tail = result[last_comma + 1:].strip()
        if tail and not tail.startswith('}') and not tail.startswith(']'):
            if ':' not in tail:
                result = result[:last_comma]

    for bracket in reversed(stack):
        if bracket == '{':
            result += '}'
        elif bracket == '[':
            result += ']'

    return result


# Compile patterns at the module level for performance
_ENTITY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("file", re.compile(r'\.(py|js|ts|ya?ml|json|md|html|css)', re.IGNORECASE)),
    ("bug", re.compile(r'\b(bug|issue|error|fix)\b|修复|问题|错误|异常', re.IGNORECASE)),
    ("tool", re.compile(r'\b(tool|cli|command|api|sdk|framework)\b|工具|命令|框架', re.IGNORECASE)),
    ("project", re.compile(r'\b(project|repo|app|application)\b|项目|仓库|应用', re.IGNORECASE)),
    ("module", re.compile(r'\b(module|component|service|layer)\b|模块|组件|服务|层', re.IGNORECASE)),
]

# 中文人名
_PERSON_CN_PATTERN = re.compile(r'^[\u4e00-\u9fa5]{2,4}$')
_PERSON_CN_BLACKLIST = {"的", "了", "是", "在", "和", "与", 
                        "模型", "算法", "函数", "系统", "测试", "模块", "组件",
                        "数据源", "缓存", "线程", "进程", "数据库"}

# 英文人名：首字母大写两段式 (如 Alice Li, John Doe)
_PERSON_EN_PATTERN = re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$')
_PERSON_EN_BLACKLIST = {
    "Web Server", "App Server", "Big Data", "Red Hat", "Domain Model", 
    "Source Code", "Test Case", "User Interface", "Virtual Machine",
    "System Design", "Machine Learning", "Deep Learning", "Data Science",
    "Code Review", "Pull Request", "Merge Request"
}

def _infer_entity_type(name: str) -> str:
    if not name.strip():
        return "concept"

    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    # 1. 规则正则匹配（顺序优先级：靠前优先）
    for entity_type, pattern in _ENTITY_PATTERNS:
        if pattern.search(name_lower):
            return entity_type

    # 2. 判断人名
    # 2.1 中文人名
    if _PERSON_CN_PATTERN.match(name_stripped):
        if not any(word in name_stripped for word in _PERSON_CN_BLACKLIST):
            return "person"
    # 2.2 英文人名
    elif _PERSON_EN_PATTERN.match(name_stripped):
        if name_stripped not in _PERSON_EN_BLACKLIST:
            return "person"

    # 3. 默认：通用技术概念
    return "concept"


# ============================================================
# Union-Find (用于实体对齐传递归一化)
# ============================================================

class UnionFind:
    """并查集: 用于实体合并的传递归一化"""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> str:
        """合并两个实体, 返回 canonical name (取较短的)"""
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            # 取较短的名字作为 canonical
            canonical = ra if len(ra) <= len(rb) else rb
            self.parent[ra] = canonical
            self.parent[rb] = canonical
            return canonical
        return ra


# ============================================================
# KGBuilder
# ============================================================

class KGBuilder:
    """
    知识图谱构建器

    流程: task summaries → LLM 三元组抽取 → 实体对齐 → NetworkX 图谱
    """

    def __init__(
        self,
        extraction_mode: Optional[str] = None,
        max_retries: int = 3,
        content_retries: int = 2,
        timeout: int = 120,
        max_tokens: int = 4000,
        merge_max_tokens: int = 200,
        merge_batch_size: int = 20,
        merge_batch_enable: bool = False,
        concurrency: int = 4,
        rate_limit_per_sec: float = 2.0,            # 新增: 线程级速率限制, 默认 2 req/s
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_dim: int = 512,
        embedding_batch_size: int = 32,
        embedding_device: str = "cpu",
        embedding_cache_folder: Optional[str] = None,
        embedding_offline_mode: bool = True,
        qdrant_path: str = "./qdrant_data",
        entities_collection: str = "entities",
        alignment_threshold: float = 0.92,
    ):
        self.max_retries = max_retries
        self.content_retries = max(1, content_retries)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.merge_max_tokens = merge_max_tokens
        self.merge_batch_size = max(1, merge_batch_size)
        self.merge_batch_enable = merge_batch_enable
        self.concurrency = max(1, concurrency)
        self.alignment_threshold = alignment_threshold
        self.entities_collection = entities_collection
        self.embedding_dim = embedding_dim
        # 线程级速率限制器, 所有 LLM 调用 (含 merge) 共享一个令牌桶
        self.rate_limiter = _LLMRateLimiter(rate_limit_per_sec)
        logger.info(f"LLM 速率限制: {rate_limit_per_sec} req/s (并发 {self.concurrency} 自动错开)")

        # 延迟到 post_init() 初始化的字段
        self._extraction_mode_param = extraction_mode  # 仅作为优先级最高的覆盖值
        self.extraction_mode: Optional[str] = None     # 由 post_init 从 config 读出最终值
        self.model: Optional[str] = None              # 由 post_init 按 extraction_mode 选定
        self.api_key: Optional[str] = None             # 由 post_init 从 opencode_models.yaml + auth.json 取
        self.base_url: Optional[str] = None            # 由 post_init 从 opencode_models.yaml 取
        self.client: Optional[OpenAI] = None           # 由 post_init 创建
        # Merge (实体合并) 独立的 model + client, 避免 reasoning 模型吃满 max_tokens
        self.merge_model: Optional[str] = None         # 由 post_init 从 config['llm.merge_model'] 读
        self.merge_base_url: Optional[str] = None
        self.merge_client: Optional[OpenAI] = None

        # Embedding 模型
        self.embedding_offline_mode = embedding_offline_mode
        if embedding_cache_folder is None:
            embedding_cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
        self.embedding_cache_folder = embedding_cache_folder

        logger.info(f"加载 embedding 模型: {embedding_model}")
        from sentence_transformers import SentenceTransformer
        try:
            self.encoder = SentenceTransformer(
                embedding_model, device=embedding_device,
                cache_folder=embedding_cache_folder,
            )
        except Exception as e:
            if embedding_offline_mode and ("not found" in str(e).lower() or "no such file" in str(e).lower()):
                logger.warning(f"本地缓存未找到, 切换到在线模式")
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self.encoder = SentenceTransformer(
                    embedding_model, device=embedding_device,
                    cache_folder=embedding_cache_folder,
                )
            else:
                raise
        logger.info("Embedding 模型加载完成")

        # Qdrant 客户端
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as rest
        self._rest = rest
        self.qdrant = QdrantClient(path=qdrant_path)

        # 确保 entities 集合存在
        self._ensure_entities_collection()

        logger.info(
            f"KGBuilder 初始化 (LLM 待 post_init 配置): concurrency={self.concurrency}, "
            f"alignment_threshold={alignment_threshold}, extraction_mode={extraction_mode!r}"
        )

    # --------------------------------------------------------
    # LLM 后初始化 (post_init)
    # --------------------------------------------------------

    def _resolve_endpoint_for_model(
        self,
        model_name: str,
        config: "Config",
    ) -> tuple[str, str]:
        """根据 model_name 从 opencode_models.yaml 查 base_url + api_key。

        Returns:
            (api_key, base_url): api_key 可能为空字符串 (需要外部再 fallback 到环境变量)
        """
        base_url = DEFAULT_FALLBACK_BASE_URL
        api_key = ""

        cfg_rel = config.get("opencode_models.config_path", "config/opencode_models.yaml")
        cfg_path = Path(cfg_rel)
        candidates: list[Path] = []
        if cfg_path.is_absolute():
            candidates.append(cfg_path)
        else:
            candidates.append(Path.cwd() / cfg_path)
            candidates.append(Path(__file__).resolve().parents[1] / cfg_rel)

        opencode_path: Optional[Path] = next((p for p in candidates if p.exists()), None)

        if opencode_path is None:
            logger.warning(
                f"找不到 opencode_models.yaml (尝试过: "
                f"{', '.join(str(p) for p in candidates)}), "
                f"模型 {model_name!r} 使用默认 base_url + 环境变量 key"
            )
            return api_key, base_url

        try:
            opencode_cfg = yaml.safe_load(opencode_path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"加载 {opencode_path} 失败: {e}, 使用默认 base_url")
            return api_key, base_url

        if not opencode_cfg:
            return api_key, base_url

        model_entry = next(
            (m for m in opencode_cfg.get("models", []) if m.get("name") == model_name),
            None,
        )
        if model_entry is None:
            logger.warning(
                f"模型 {model_name!r} 不在 {opencode_path} 中, "
                f"使用默认 base_url + 环境变量 key"
            )
            return api_key, base_url

        endpoint_name = model_entry.get("endpoint")
        endpoint = opencode_cfg.get("endpoints", {}).get(endpoint_name or "", {})
        if endpoint.get("base_url"):
            base_url = endpoint["base_url"]

        auth_file_rel = opencode_cfg.get("auth_file", "~/.local/share/opencode/auth.json")
        auth_file = Path(auth_file_rel).expanduser()
        if auth_file.exists():
            try:
                auth_data = json.loads(auth_file.read_text(encoding="utf-8"))
                auth_provider = endpoint.get("auth_provider", "opencode-go")
                api_key = (auth_data.get(auth_provider, {}).get("key", "") or "")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"读取 {auth_file} 失败: {e}")
        else:
            logger.warning(f"auth.json 不存在: {auth_file}")

        return api_key, base_url

    def _init_client(
        self,
        model_name: str,
        config: "Config",
    ) -> tuple[Optional[str], Optional[str], Optional[OpenAI]]:
        """根据 model_name 解析 endpoint 并创建 OpenAI client。

        Returns:
            (api_key, base_url, client) 任一为 None 表示初始化失败
        """
        api_key, base_url = self._resolve_endpoint_for_model(model_name, config)
        if not api_key:
            api_key = os.environ.get(DEFAULT_FALLBACK_API_KEY_ENV, "")
        if not api_key:
            logger.warning(
                f"模型 {model_name!r}: 未找到 API Key (opencode_models.yaml + 环境变量都无),"
                f" 该 client 初始化失败"
            )
            return None, base_url, None
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout)
        return api_key, base_url, client

    def post_init(self, config: "Config") -> None:
        """后初始化: 在 KGBuilder() 之后由调用方显式调用。

        职责 (按顺序):
          1. 若 __init__ 的 extraction_mode 参数被显式传入 (非 None),
             覆盖 config['knowledge_graph.extraction_mode'] (优先级最高)。
          2. 从 config 读出最终 extraction_mode,
             选 self.model = config['llm.triple_model'] (triple 模式)
                    或 config['llm.entity_model'] (entity 模式),
                    缺省时回退到 config['llm.model']。
          3. 用 _init_client 解析 endpoint, 创建 self.client。
          4. 用 _init_client 解析 merge endpoint, 创建 self.merge_client
             (model = config['llm.merge_model'], 缺省回退到 self.model)。

        若 opencode_models.yaml 中未列出 self.model, 或文件/auth.json 缺失,
        回退到默认 Zen 端点 + OPENCODE_ZEN_API_KEY 环境变量, 仍失败则抛 ValueError。
        """
        # ── 1. extraction_mode 覆盖 (优先级最高) ──
        if self._extraction_mode_param is not None:
            config.set("knowledge_graph.extraction_mode", self._extraction_mode_param)

        # ── 2. 选 main model ──
        self.extraction_mode = config.get("knowledge_graph.extraction_mode", "triple")
        fallback_model = config.get("llm.model", "deepseek-v4-flash-free")
        if self.extraction_mode == "entity":
            self.model = config.get("llm.entity_model") or fallback_model
        else:
            self.model = config.get("llm.triple_model") or fallback_model

        # ── 3. 解析并创建 main client ──
        self.api_key, self.base_url, self.client = self._init_client(self.model, config)
        if self.client is None:
            raise ValueError(
                f"未找到 API Key: opencode_models.yaml 加载失败或不含模型 {self.model!r}, "
                f"且环境变量 {DEFAULT_FALLBACK_API_KEY_ENV} 未设置。\n"
                f"请检查: (1) opencode_models.yaml 是否包含模型 {self.model}; "
                f"(2) auth.json 是否存在并包含对应 provider; "
                f"(3) export {DEFAULT_FALLBACK_API_KEY_ENV}='your-key'"
            )

        # ── 4. 解析并创建 merge client (独立于 main) ──
        self.merge_model = config.get("llm.merge_model") or self.model
        if self.merge_model == self.model:
            # merge 没单独配, 直接复用 main client (避免重复建连)
            self.merge_base_url = self.base_url
            self.merge_client = self.client
            logger.info(
                f"merge_model 未独立配置, 复用 main model: {self.merge_model}"
            )
        else:
            self.merge_api_key, self.merge_base_url, self.merge_client = self._init_client(
                self.merge_model, config
            )
            if self.merge_client is None:
                raise ValueError(
                    f"merge_model {self.merge_model!r} 初始化失败: "
                    f"opencode_models.yaml 不含此模型, 且环境变量未设。"
                )

        logger.info(
            f"KGBuilder LLM 后初始化完成: model={self.model}, base_url={self.base_url}, "
            f"extraction_mode={self.extraction_mode}, "
            f"merge_model={self.merge_model}, merge_base_url={self.merge_base_url}, "
            f"concurrency={self.concurrency}"
        )

    # --------------------------------------------------------
    # Qdrant entities 集合
    # --------------------------------------------------------

    def _ensure_entities_collection(self):
        """确保 entities 集合存在"""
        rest = self._rest
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.entities_collection not in collections:
            self.qdrant.create_collection(
                collection_name=self.entities_collection,
                vectors_config=rest.VectorParams(
                    size=self.embedding_dim,
                    distance=rest.Distance.COSINE,
                ),
            )
            logger.info(f"创建集合: {self.entities_collection} (dim={self.embedding_dim})")
        else:
            logger.info(f"集合已存在: {self.entities_collection}")

    # --------------------------------------------------------
    # LLM 调用
    # --------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        *,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
        thinking: Optional[dict] = None,
    ) -> str:
        """调用 LLM 获取响应。

        默认使用 self.model + self.client; 也可显式传入 (用于 merge 任务用独立 client)。
        thinking: 透传 extra_body={"thinking": ...}, 例如 {"type": "disabled"}。
        """
        use_model = model or self.model
        use_client = client or self.client
        if use_client is None or use_model is None:
            raise RuntimeError(
                "KGBuilder 尚未完成 LLM 初始化, 请先调用 builder.post_init(config)。"
            )
        tokens = max_tokens or self.max_tokens
        kwargs = {
            "model": use_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": tokens,
        }
        if thinking is not None:
            kwargs["extra_body"] = {"thinking": thinking}
        response = use_client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        content = msg.content

        # Reasoning 模型 fallback: content 为空时从 reasoning_content 提取 JSON
        if not content or not content.strip():
            reasoning = None
            if hasattr(msg, 'model_extra') and msg.model_extra:
                reasoning = msg.model_extra.get('reasoning_content', '')
            if reasoning and reasoning.strip():
                logger.warning("LLM content 为空, 尝试从 reasoning_content 提取 JSON")
                extracted = _extract_json_from_response(reasoning)
                # 验证提取结果是否包含有效 JSON
                try:
                    json.loads(extracted)
                    content = extracted
                    logger.info("从 reasoning_content 成功提取 JSON")
                except json.JSONDecodeError:
                    # 提取失败, 直接使用 reasoning_content 原始内容
                    logger.warning("reasoning_content 中未找到有效 JSON, 使用原始内容")
                    content = reasoning
            else:
                raise ValueError("LLM 返回空内容")

        return content.strip()

    def _call_llm_with_retry(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        *,
        model: Optional[str] = None,
        client: Optional[OpenAI] = None,
        thinking: Optional[dict] = None,
    ) -> str:
        """带指数退避重试的 LLM 调用 (model/client/thinking 可选, 透传给 _call_llm)。

        - 调用前先 acquire 速率限制令牌 (rate_limiter), 多线程自动错开
        - 429 / rate_limit_error / FreeUsageLimitError 走长退避 (30s 起步, 指数: 30/60/120)
        - 其他错误走短退避 (2s 起步, 指数: 2/4/8)
        """
        last_error = None
        for attempt in range(self.max_retries):
            # 每次重试前都重新 acquire 一次令牌
            self.rate_limiter.acquire()
            try:
                return self._call_llm(
                    prompt,
                    max_tokens,
                    model=model,
                    client=client,
                    thinking=thinking,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # 区分 rate_limit 类错误, 走不同的退避策略
                    err_str = str(e).lower()
                    is_rate_limit = (
                        "rate_limit" in err_str
                        or "429" in err_str
                        or "2062" in err_str
                        or "freeusagelimit" in err_str
                        or "ratelimitexceeded" in err_str
                    )
                    if is_rate_limit:
                        # 30s 起步, 指数退避: 30 / 60 / 120
                        wait_time = 30 * (2 ** attempt)
                    else:
                        # 2s 起步, 指数退避: 2 / 4 / 8
                        wait_time = 2 ** (attempt + 1)
                    err_kind = "rate_limit" if is_rate_limit else "其他错误"
                    logger.warning(
                        f"LLM 调用失败 (尝试 {attempt + 1}/{self.max_retries}, {err_kind}): {e}, "
                        f"{wait_time}s 后重试"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM 调用失败, 已达最大重试: {e}")
        raise last_error  # type: ignore

    # --------------------------------------------------------
    # 三元组抽取
    # --------------------------------------------------------

    def extract_triples(self, task: Task) -> list[Triple]:
        """从单个 task 的 summary 中抽取三元组"""
        prompt = load_prompt("TRIPLE_EXTRACTION_PROMPT").format(task_summary=task.task_summary)

        for content_attempt in range(self.content_retries):
            try:
                output = self._call_llm_with_retry(prompt)
                triples = self._parse_triples(output, task.task_id)

                if not triples:
                    if content_attempt < self.content_retries - 1:
                        logger.warning(
                            f"Task {task.task_id}: 三元组为空, "
                            f"重试 ({content_attempt + 1}/{self.content_retries})"
                        )
                        continue
                    logger.warning(f"Task {task.task_id}: 未能抽取到三元组")
                    return []

                return triples

            except Exception as e:
                logger.error(f"Task {task.task_id}: 抽取失败: {e}")
                if content_attempt < self.content_retries - 1:
                    continue
                return []

        return []

    def _parse_triples(self, llm_output: str, source_task_id: str) -> list[Triple]:
        """解析 LLM 输出的三元组 JSON"""
        json_str = _extract_json_from_response(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败, 尝试修复")
            repaired = _repair_truncated_json(json_str)
            try:
                data = json.loads(repaired)
                logger.info("JSON 修复成功")
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败且无法修复: {e}")
                return []

        triples_data = data.get("triples", [])
        if not isinstance(triples_data, list):
            logger.error(f"triples 应为列表, 实际: {type(triples_data)}")
            return []

        triples = []
        for t in triples_data:
            head = str(t.get("head", "")).strip()
            relation = str(t.get("relation", "")).strip()
            tail = str(t.get("tail", "")).strip()
            confidence = float(t.get("confidence", 0.5))

            # 校验: head/relation/tail 非空
            if not head or not relation or not tail:
                logger.debug(f"跳过空三元组: ({head}, {relation}, {tail})")
                continue

            # 过滤过于通用的实体
            generic = {"用户", "代码", "系统", "问题", "方法", "方式", "过程", "结果"}
            if head in generic or tail in generic:
                logger.debug(f"跳过通用实体三元组: ({head}, {relation}, {tail})")
                continue

            triples.append(Triple(
                head=head,
                relation=relation,
                tail=tail,
                confidence=min(max(confidence, 0.0), 1.0),
                source_task_id=source_task_id,
            ))

        return triples

    def extract_all_triples(
        self,
        tasks: list[Task],
        output_file: Optional[str] = None,
        force: bool = False,
    ) -> list[Triple]:
        """
        批量抽取所有 task 的三元组

        Args:
            tasks: task 列表
            output_file: 增量输出 JSONL 路径 (每个 task 抽取后立即追加)
            force: 强制重抽。True 时会先删除 output_file (如有), 然后从零开始,
                   跳过断点续传逻辑。

        注意:
            默认会从 output_file 恢复已有进度——若文件中所有 task_id 都已存在,
            则不会调用 LLM, 直接复用旧结果。如需重跑, 请传 force=True
            或在 CLI 加 --force 参数。
        """
        all_triples: list[Triple] = []
        total = len(tasks)

        # 加载已有进度 (断点续传)
        done_task_ids: set[str] = set()
        if output_file and Path(output_file).exists():
            if force:
                # 强制模式: 先删旧文件, 跳过恢复
                Path(output_file).unlink(missing_ok=True)
                logger.warning(
                    f"[force] 已删除旧抽取文件 {output_file}, 重新从零开始抽取三元组"
                )
            else:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            t = Triple.from_dict(json.loads(line))
                            all_triples.append(t)
                            done_task_ids.add(t.source_task_id)
                logger.warning(
                    f"检测到 {output_file} 已有 {len(all_triples)} 条三元组 / "
                    f"{len(done_task_ids)} 个 task, 将复用旧结果。"
                    f"如需重新抽取, 请加 --force 参数。"
                )

        # 过滤已完成的 task
        pending_tasks = [t for t in tasks if t.task_id not in done_task_ids]
        if not pending_tasks:
            logger.warning(
                f"所有 {len(tasks)} 个 task 在抽取结果中均已存在, "
                f"本次未调用 LLM, 直接复用旧结果。"
                f"如需重抽请加 --force。"
            )
            return all_triples

        logger.info(f"开始三元组抽取: {len(pending_tasks)}/{total} 个 task 待处理, "
                    f"concurrency={self.concurrency}")

        def _process_task(task: Task) -> tuple[str, list[Triple]]:
            triples = self.extract_triples(task)
            return task.task_id, triples

        # 并发抽取
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(_process_task, task): task
                for task in pending_tasks
            }

            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                completed += 1

                try:
                    task_id, triples = future.result()
                    all_triples.extend(triples)

                    # 增量写入
                    if output_file:
                        with open(output_file, "a", encoding="utf-8") as f:
                            for t in triples:
                                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

                    logger.info(
                        f"[{completed}/{len(pending_tasks)}] {task.task_label}: "
                        f"{len(triples)} 个三元组"
                    )

                except Exception as e:
                    logger.error(f"[{completed}/{len(pending_tasks)}] "
                               f"{task.task_label}: 失败 - {e}")

        logger.info(f"三元组抽取完成: 共 {len(all_triples)} 个三元组")
        return all_triples

    # --------------------------------------------------------
    # 直接实体提取 (entity 模式)
    # --------------------------------------------------------

    def extract_entities(self, task: Task) -> list[str]:
        """从单个 task 的 summary 中直接提取关键实体"""
        prompt = load_prompt("ENTITY_EXTRACTION_PROMPT").format(task_summary=task.task_summary)

        for content_attempt in range(self.content_retries):
            try:
                output = self._call_llm_with_retry(prompt)
                entities = self._parse_entities(output)

                if not entities:
                    if content_attempt < self.content_retries - 1:
                        logger.warning(
                            f"Task {task.task_id}: 实体为空, "
                            f"重试 ({content_attempt + 1}/{self.content_retries})"
                        )
                        continue
                    logger.warning(f"Task {task.task_id}: 未能提取到实体")
                    return []

                return entities

            except Exception as e:
                logger.error(f"Task {task.task_id}: 实体提取失败: {e}")
                if content_attempt < self.content_retries - 1:
                    continue
                return []

        return []

    def _parse_entities(self, llm_output: str) -> list[str]:
        """解析 LLM 输出的实体列表 JSON"""
        json_str = _extract_json_from_response(llm_output)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(json_str)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError as e:
                logger.error(f"实体 JSON 解析失败且无法修复: {e}")
                return []

        # 支持 {"entities": [...]} 和直接 [...] 两种格式
        if isinstance(data, dict):
            entities_data = data.get("entities", [])
        elif isinstance(data, list):
            entities_data = data
        else:
            logger.error(f"entities 应为列表或对象, 实际: {type(data)}")
            return []

        # 过滤通用词和空值
        generic = {"用户", "代码", "系统", "问题", "方法", "方式", "过程", "结果"}
        entities = []
        for e in entities_data:
            name = str(e).strip()
            if name and name not in generic:
                entities.append(name)

        return entities

    def extract_all_entities(
        self,
        tasks: list[Task],
        output_file: Optional[str] = None,
        force: bool = False,
    ) -> dict[str, list[str]]:
        """
        批量提取所有 task 的关键实体, 构建倒排索引

        Args:
            tasks: task 列表
            output_file: 增量输出 JSONL 路径
            force: 强制重抽。True 时会先删除 output_file (如有), 然后从零开始,
                   跳过断点续传逻辑。

        Returns:
            inverted_index: {entity_name: [task_ids]}

        注意:
            默认会从 output_file 恢复已有进度——若文件中所有 task_id 都已存在,
            则不会调用 LLM, 直接复用旧结果。如需重跑, 请传 force=True
            或在 CLI 加 --force 参数。
        """
        inverted_index: dict[str, list[str]] = {}
        total = len(tasks)

        # 加载已有进度 (断点续传)
        done_task_ids: set[str] = set()
        if output_file and Path(output_file).exists():
            if force:
                # 强制模式: 先删旧文件, 跳过恢复
                Path(output_file).unlink(missing_ok=True)
                logger.warning(
                    f"[force] 已删除旧抽取文件 {output_file}, 重新从零开始提取实体"
                )
            else:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                task_id = record.get("task_id", "")
                                entities = record.get("entities", [])
                                done_task_ids.add(task_id)
                                for entity in entities:
                                    if entity not in inverted_index:
                                        inverted_index[entity] = []
                                    if task_id not in inverted_index[entity]:
                                        inverted_index[entity].append(task_id)
                            except json.JSONDecodeError:
                                continue
                logger.warning(
                    f"检测到 {output_file} 已有 {len(inverted_index)} 个实体 / "
                    f"{len(done_task_ids)} 个 task, 将复用旧结果。"
                    f"如需重新提取, 请加 --force 参数。"
                )

        # 过滤已完成的 task
        pending_tasks = [t for t in tasks if t.task_id not in done_task_ids]
        if not pending_tasks:
            logger.warning(
                f"所有 {len(tasks)} 个 task 在抽取结果中均已存在, "
                f"本次未调用 LLM, 直接复用旧结果。"
                f"如需重抽请加 --force。"
            )
            return inverted_index

        logger.info(
            f"开始实体提取: {len(pending_tasks)}/{total} 个 task 待处理, "
            f"concurrency={self.concurrency}"
        )

        def _process_task(task: Task) -> tuple[str, list[str]]:
            entities = self.extract_entities(task)
            return task.task_id, entities

        # 并发提取
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {
                executor.submit(_process_task, task): task
                for task in pending_tasks
            }

            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                completed += 1

                try:
                    task_id, entities = future.result()

                    # 更新倒排索引
                    for entity in entities:
                        if entity not in inverted_index:
                            inverted_index[entity] = []
                        if task_id not in inverted_index[entity]:
                            inverted_index[entity].append(task_id)

                    # 增量写入
                    if output_file:
                        with open(output_file, "a", encoding="utf-8") as f:
                            record = {"task_id": task_id, "entities": entities}
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    logger.info(
                        f"[{completed}/{len(pending_tasks)}] {task.task_label}: "
                        f"{len(entities)} 个实体"
                    )

                except Exception as e:
                    logger.error(
                        f"[{completed}/{len(pending_tasks)}] "
                        f"{task.task_label}: 失败 - {e}"
                    )

        logger.info(
            f"实体提取完成: {len(inverted_index)} 个唯一实体, "
            f"覆盖 {len(done_task_ids) + completed} 个 task"
        )
        return inverted_index

    def collect_entities_from_index(
        self,
        inverted_index: dict[str, list[str]],
    ) -> dict[str, Entity]:
        """从倒排索引构建 entity_map"""
        entity_map: dict[str, Entity] = {}
        for name, task_ids in inverted_index.items():
            entity_map[name] = Entity(
                name=name,
                entity_type=_infer_entity_type(name),
                source_tasks=list(task_ids),
            )
        logger.info(f"从倒排索引收集到 {len(entity_map)} 个唯一实体")
        return entity_map

    # --------------------------------------------------------
    # 实体对齐
    # --------------------------------------------------------

    def collect_entities(self, triples: list[Triple]) -> dict[str, Entity]:
        """从三元组中收集所有唯一实体"""
        entity_map: dict[str, Entity] = {}

        for t in triples:
            for name in [t.head, t.tail]:
                if name not in entity_map:
                    entity_map[name] = Entity(
                        name=name,
                        entity_type=_infer_entity_type(name),
                        source_tasks=[t.source_task_id],
                    )
                else:
                    if t.source_task_id not in entity_map[name].source_tasks:
                        entity_map[name].source_tasks.append(t.source_task_id)

        logger.info(f"收集到 {len(entity_map)} 个唯一实体")
        return entity_map

    def align_entities(
        self,
        entity_map: dict[str, Entity],
    ) -> dict[str, str]:
        """
        实体对齐: Qdrant 向量检索 + LLM 批量二次确认

        优化:
        - 增量 embedding: 已存在于 Qdrant 的实体复用已有向量, 只对新增实体编码
        - 批量 LLM 确认: 将候选合并对分批发送给 LLM, 减少请求次数

        Returns:
            canonical_map: {原始名: canonical名}
        """
        entities = list(entity_map.keys())
        if len(entities) < 2:
            logger.info("实体数 < 2, 跳过对齐")
            return {name: name for name in entities}

        # 1. 增量 Embedding: 区分已有 / 新增实体
        existing_entities = []  # 已在 Qdrant 中的实体名
        new_entities = []       # 需要新 embed 的实体名
        cached_vectors = {}     # {name: vector_list}

        check_ids = [_stable_uuid(name) for name in entities]
        try:
            existing_points = self.qdrant.retrieve(
                collection_name=self.entities_collection,
                ids=check_ids,
                with_vectors=True,
            )
            existing_ids = {p.id for p in existing_points}
            for p in existing_points:
                name = (p.payload or {}).get("name", "")
                if name:
                    cached_vectors[name] = p.vector
        except Exception as e:
            logger.warning(f"Qdrant retrieve 失败, 回退到全量 embedding: {e}")
            existing_ids = set()

        for name in entities:
            if _stable_uuid(name) in existing_ids and name in cached_vectors:
                existing_entities.append(name)
            else:
                new_entities.append(name)

        logger.info(
            f"增量 embedding: {len(existing_entities)} 个已有 (复用向量), "
            f"{len(new_entities)} 个新增"
        )

        # 只 encode 新实体
        if new_entities:
            logger.info(f"Embedding {len(new_entities)} 个新增实体")
            new_vectors = self.encoder.encode(
                new_entities,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            for name, vec in zip(new_entities, new_vectors):
                cached_vectors[name] = vec.tolist()

        # 按原始顺序组装所有向量
        vectors = [cached_vectors[name] for name in entities]
        logger.info(f"实体向量就绪: {len(vectors)} 个")

        # 2. Upsert 所有实体 (更新 payload: source_tasks 可能变化)
        rest = self._rest
        points = []
        for name, vec in zip(entities, vectors):
            point_id = _stable_uuid(name)
            ent = entity_map[name]
            points.append(rest.PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "name": name,
                    "entity_type": ent.entity_type,
                    "source_tasks": ent.source_tasks,
                },
            ))

        upsert_batch = 100
        for i in range(0, len(points), upsert_batch):
            batch = points[i:i + upsert_batch]
            self.qdrant.upsert(
                collection_name=self.entities_collection,
                points=batch,
            )
        logger.info(f"已 upsert {len(points)} 个实体到 Qdrant")

        # 3. 逐实体检索相似实体 (本地矩阵乘法, 替代 Qdrant round trip)
        vec_matrix = np.asarray(vectors, dtype=np.float32)
        sim_matrix = vec_matrix @ vec_matrix.T
        n = len(entities)
        merge_candidates: dict[tuple[str, str], float] = {}
        for i in range(n):
            sims = sim_matrix[i].copy()
            sims[i] = 0.0
            above = np.where(sims >= self.alignment_threshold)[0]
            for j in above:
                if j <= i:
                    continue
                pair = tuple(sorted([entities[i], entities[j]]))
                score = float(sims[j])
                if score > merge_candidates.get(pair, 0.0):
                    merge_candidates[pair] = score

        logger.info(f"发现 {len(merge_candidates)} 对候选合并实体 (threshold={self.alignment_threshold})")

        if not merge_candidates:
            return {name: name for name in entities}

        # 4. LLM 二次确认 (批量 / 单条)
        # 先打印实体对齐使用的模型, 方便排查 "为啥合并/不合并"
        merge_mode = "批量" if self.merge_batch_enable else "单条"
        if self.merge_model == self.model and self.merge_client is self.client:
            merge_model_note = f"{self.merge_model} (复用 main client)"
        else:
            merge_model_note = (
                f"{self.merge_model} (独立 client, base_url={self.merge_base_url})"
            )
        logger.info(
            f"实体对齐 LLM 确认: {merge_mode}模式, {len(merge_candidates)} 对候选, "
            f"使用模型 {merge_model_note}"
        )

        uf = UnionFind()
        for name in entities:
            uf.find(name)  # 初始化

        confirmed_pairs = 0
        pair_list = list(merge_candidates.keys())

        if self.merge_batch_enable:
            # ---- 批量模式: 每批发送多对, 减少 API 调用 ----
            total_batches = (len(pair_list) + self.merge_batch_size - 1) // self.merge_batch_size

            for batch_idx in range(0, len(pair_list), self.merge_batch_size):
                batch = pair_list[batch_idx:batch_idx + self.merge_batch_size]

                # 过滤已在同一组的对
                need_check = [(e1, e2) for e1, e2 in batch if uf.find(e1) != uf.find(e2)]
                if not need_check:
                    continue

                batch_num = batch_idx // self.merge_batch_size + 1
                logger.info(
                    f"  批量确认 [{batch_num}/{total_batches}]: "
                    f"{len(need_check)} 对待确认"
                )

                results = self._batch_llm_confirm_merges(need_check, entity_map)

                for (e1, e2), should_merge in results.items():
                    if should_merge and uf.find(e1) != uf.find(e2):
                        canonical = uf.union(e1, e2)
                        confirmed_pairs += 1
                        score = merge_candidates.get((e1, e2),
                                    merge_candidates.get((e2, e1), 0.0))
                        logger.info(f"  合并: '{e1}' + '{e2}' → '{canonical}' (score={score:.4f})")
        else:
            # ---- 单条模式: 逐条确认, 结果更稳定 ----
            logger.info(f"  单条确认模式: {len(pair_list)} 对候选")

            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {
                    pool.submit(self._llm_confirm_merge, e1, e2, entity_map): (e1, e2)
                    for e1, e2 in pair_list
                }
                for future in as_completed(futures):
                    e1, e2 = futures[future]
                    try:
                        should_merge = future.result()
                    except Exception as e:
                        logger.warning(f"  单条确认异常: '{e1}' vs '{e2}': {e}")
                        continue

                    if should_merge and uf.find(e1) != uf.find(e2):
                        canonical = uf.union(e1, e2)
                        confirmed_pairs += 1
                        score = merge_candidates.get((e1, e2),
                                    merge_candidates.get((e2, e1), 0.0))
                        logger.info(f"  合并: '{e1}' + '{e2}' → '{canonical}' (score={score:.4f})")

        # 5. 构建 canonical map
        canonical_map = {}
        for name in entities:
            canonical_map[name] = uf.find(name)

        unique_count = len(set(canonical_map.values()))
        logger.info(
            f"实体对齐完成: {len(entities)} → {unique_count} 个唯一实体, "
            f"合并 {confirmed_pairs} 对"
        )
        return canonical_map

    def _llm_confirm_merge(self, entity_a: str, entity_b: str,
                           entity_map: Optional[dict[str, Entity]] = None) -> bool:
        """LLM 确认两个实体是否应该合并 (用独立的 merge_model + merge_client)"""
        # 构建上下文信息
        context_a = f"实体类型: {_infer_entity_type(entity_a)}"
        context_b = f"实体类型: {_infer_entity_type(entity_b)}"
        if entity_map:
            ea = entity_map.get(entity_a)
            eb = entity_map.get(entity_b)
            if ea and ea.source_tasks:
                context_a += f", 来源task数: {len(ea.source_tasks)}"
            if eb and eb.source_tasks:
                context_b += f", 来源task数: {len(eb.source_tasks)}"

        prompt = load_prompt("ENTITY_MERGE_PROMPT").format(
            entity_a=entity_a, entity_b=entity_b,
            context_a=context_a, context_b=context_b,
        )
        try:
            output = self._call_llm_with_retry(
                prompt,
                max_tokens=self.merge_max_tokens,
                model=self.merge_model,
                client=self.merge_client,
                thinking={"type": "disabled"},
            )
            output_upper = output.strip().upper()
            return "MERGE" in output_upper
        except Exception as e:
            logger.warning(f"实体合并确认失败 ({entity_a}, {entity_b}): {e}")
            return False

    def _build_merge_context(self, name: str, entity_map: dict[str, Entity]) -> str:
        """为批量合并 prompt 构建实体上下文描述"""
        etype = _infer_entity_type(name)
        ctx = f"{name} (类型: {etype}"
        ent = entity_map.get(name)
        if ent and ent.source_tasks:
            ctx += f", 来源task数: {len(ent.source_tasks)}"
        ctx += ")"
        return ctx

    def _parse_batch_merge_response(
        self,
        raw: str,
        pairs: list[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], bool], int]:
        """解析批量合并 LLM 响应

        Returns:
            (results, total_decisions): results 是 {(e1,e2): should_merge},
            total_decisions 是成功解析的决策条数 (0 表示解析失败)
        """
        results = {pair: False for pair in pairs}

        # 优先直接解析 (处理 JSON 数组)
        decisions = None
        raw_stripped = raw.strip()

        # 尝试 1: 直接 json.loads
        try:
            decisions = json.loads(raw_stripped)
        except json.JSONDecodeError:
            pass

        # 尝试 2: _extract_json_from_response (处理 markdown/嵌套等)
        if decisions is None:
            json_str = _extract_json_from_response(raw)
            try:
                decisions = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 尝试 3: raw_decode 从第一个 [ 开始
        if decisions is None:
            first_bracket = raw_stripped.find("[")
            if first_bracket != -1:
                decoder = json.JSONDecoder()
                try:
                    decisions, _ = decoder.raw_decode(raw_stripped, first_bracket)
                except json.JSONDecodeError:
                    pass

        # 尝试 4: _repair_truncated_json
        if decisions is None:
            json_str = _extract_json_from_response(raw)
            repaired = _repair_truncated_json(json_str)
            try:
                decisions = json.loads(repaired)
            except json.JSONDecodeError as e:
                logger.error(f"批量合并响应 JSON 解析失败: {e}")
                return results, 0

        if not isinstance(decisions, list):
            # LLM 可能返回 dict 包裹列表, 如 {"decisions": [...]}
            if isinstance(decisions, dict):
                for v in decisions.values():
                    if isinstance(v, list):
                        decisions = v
                        break
                else:
                    logger.error(f"批量合并响应 dict 中未找到列表值")
                    return results, 0
            else:
                logger.error(f"批量合并响应应为列表, 实际: {type(decisions)}")
                return results, 0

        # 按 (sorted pair) 匹配
        pair_set = {tuple(sorted(p)): p for p in pairs}
        matched = 0
        for d in decisions:
            if not isinstance(d, dict):
                continue
            pair_raw = d.get("pair", [])
            action = str(d.get("action", "KEEP")).strip().upper()
            if isinstance(pair_raw, list) and len(pair_raw) == 2:
                key = tuple(sorted([str(pair_raw[0]), str(pair_raw[1])]))
                if key in pair_set:
                    results[pair_set[key]] = (action == "MERGE")
                    matched += 1

        # Fallback: LLM 返回扁平字符串列表 (如 ["A", "B", "C", "D"])
        # 将连续两项视为一对合并决策
        if matched == 0 and all(isinstance(d, str) for d in decisions):
            logger.info(f"批量合并 fallback: 扁平列表 {len(decisions)} 项")
            all_entity_names = set()
            for e1, e2 in pairs:
                all_entity_names.add(e1)
                all_entity_names.add(e2)

            # 策略1: 连续配对 (0,1), (2,3), ...
            i = 0
            while i + 1 < len(decisions):
                key = tuple(sorted([decisions[i], decisions[i+1]]))
                if key in pair_set:
                    results[pair_set[key]] = True
                    matched += 1
                i += 2

            # 策略2: 如果连续配对也没匹配上, 视为"应合并的实体集合"
            if matched == 0:
                merge_names = {s for s in decisions if s in all_entity_names}
                if merge_names:
                    logger.info(f"批量合并 fallback: 视为合并集合 {merge_names}")
                    for pair_key, pair_orig in pair_set.items():
                        if pair_key[0] in merge_names and pair_key[1] in merge_names:
                            results[pair_orig] = True
                            matched += 1

        if matched < len(pairs):
            logger.warning(
                f"批量合并响应: {matched}/{len(pairs)} 对匹配, "
                f"未匹配的默认 KEEP"
            )
        return results, matched

    def _batch_llm_confirm_merges(
        self,
        pairs: list[tuple[str, str]],
        entity_map: dict[str, Entity],
    ) -> dict[tuple[str, str], bool]:
        """批量 LLM 确认实体合并

        Args:
            pairs: 待确认的 (entity_a, entity_b) 列表
            entity_map: 实体映射表 (提供上下文)

        Returns:
            {(entity_a, entity_b): should_merge}
        """
        if not pairs:
            return {}

        # 构建 prompt 中的实体对描述
        lines = []
        for i, (e1, e2) in enumerate(pairs):
            ctx1 = self._build_merge_context(e1, entity_map)
            ctx2 = self._build_merge_context(e2, entity_map)
            lines.append(f"{i+1}. [{e1}] vs [{e2}]  —  A: {ctx1}  B: {ctx2}")

        prompt = load_prompt("BATCH_ENTITY_MERGE_PROMPT").format(pairs_text="\n".join(lines))

        # max_tokens 按 batch 大小动态调整
        batch_tokens = max(self.merge_batch_size * 40, 500)
        content_retries = 3
        best_results: dict[tuple[str, str], bool] = {}
        best_decisions = 0

        for attempt in range(content_retries):
            try:
                output = self._call_llm_with_retry(
                    prompt,
                    max_tokens=batch_tokens,
                    model=self.merge_model,
                    client=self.merge_client,
                    thinking={"type": "disabled"},
                )
                results, total_decisions = self._parse_batch_merge_response(output, pairs)

                # 保留最佳结果
                if total_decisions > best_decisions:
                    best_results = results
                    best_decisions = total_decisions

                # 全部解析成功, 直接使用
                if total_decisions >= len(pairs):
                    return results

                if attempt < content_retries - 1:
                    logger.warning(
                        f"批量合并响应: {total_decisions}/{len(pairs)} 对解析, "
                        f"重试 ({attempt + 1}/{content_retries})"
                    )

            except Exception as e:
                logger.warning(f"批量合并 LLM 调用失败 (尝试 {attempt + 1}): {e}")

        undecided = [p for p in pairs if best_results.get(p) is None]

        if undecided:
            if best_decisions == 0:
                logger.warning(
                    f"批量合并确认失败 ({len(undecided)} 对), 降级为单条确认"
                )
            else:
                logger.info(
                    f"批量未决 {len(undecided)}/{len(pairs)} 对, 补充单条确认"
                )
            for e1, e2 in undecided:
                best_results[(e1, e2)] = self._llm_confirm_merge(e1, e2, entity_map)

        return best_results

    # --------------------------------------------------------
    # 图谱构建
    # --------------------------------------------------------

    def build_graph(
        self,
        triples: list[Triple],
        canonical_map: dict[str, str],
        entity_map: dict[str, Entity],
    ) -> nx.MultiDiGraph:
        """构建 NetworkX MultiDiGraph"""
        G = nx.MultiDiGraph()

        # 1. 添加节点 (canonical entities)
        canonical_entities: dict[str, dict] = {}
        for name, canonical in canonical_map.items():
            if canonical not in canonical_entities:
                ent = entity_map.get(name)
                entity_type = ent.entity_type if ent else _infer_entity_type(canonical)
                canonical_entities[canonical] = {
                    "type": entity_type,
                    "aliases": [],
                    "source_tasks": [],
                }
            # 收集别名和来源 task
            info = canonical_entities[canonical]
            if name != canonical:
                info["aliases"].append(name)
            ent = entity_map.get(name)
            if ent:
                for tid in ent.source_tasks:
                    if tid not in info["source_tasks"]:
                        info["source_tasks"].append(tid)

        for canonical, info in canonical_entities.items():
            G.add_node(
                canonical,
                entity_type=info["type"],
                aliases=info["aliases"],
                source_tasks=info["source_tasks"],
            )

        # 2. 添加边
        for triple in triples:
            head = canonical_map.get(triple.head, triple.head)
            tail = canonical_map.get(triple.tail, triple.tail)

            # 确保节点存在
            if head not in G:
                G.add_node(head, entity_type=_infer_entity_type(head), aliases=[], source_tasks=[])
            if tail not in G:
                G.add_node(tail, entity_type=_infer_entity_type(tail), aliases=[], source_tasks=[])

            G.add_edge(
                head, tail,
                relation=triple.relation,
                weight=triple.confidence,
                source_task=triple.source_task_id,
            )

        logger.info(
            f"图谱构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边"
        )
        return G

    def build_cooccurrence_graph(
        self,
        inverted_index: dict[str, list[str]],
        canonical_map: dict[str, str],
        entity_map: dict[str, Entity],
    ) -> nx.MultiDiGraph:
        """
        从倒排索引构建共现图谱

        节点 = canonical 实体
        边 = 两个 canonical 实体在同一 task 中共现

        Args:
            inverted_index: {entity_name: [task_ids]}
            canonical_map: {原始名: canonical名}
            entity_map: {原始名: Entity}
        """
        from itertools import combinations

        G = nx.MultiDiGraph()

        # 1. 添加节点 (canonical entities)
        canonical_entities: dict[str, dict] = {}
        for name, canonical in canonical_map.items():
            if canonical not in canonical_entities:
                ent = entity_map.get(name)
                entity_type = ent.entity_type if ent else _infer_entity_type(canonical)
                canonical_entities[canonical] = {
                    "type": entity_type,
                    "aliases": [],
                    "source_tasks": [],
                }
            info = canonical_entities[canonical]
            if name != canonical:
                info["aliases"].append(name)
            ent = entity_map.get(name)
            if ent:
                for tid in ent.source_tasks:
                    if tid not in info["source_tasks"]:
                        info["source_tasks"].append(tid)

        for canonical, info in canonical_entities.items():
            G.add_node(
                canonical,
                entity_type=info["type"],
                aliases=info["aliases"],
                source_tasks=info["source_tasks"],
            )

        # 2. 构建共现边: 同一 task 中的实体两两配对
        # 先构建 task → [canonical entities] 映射
        task_entities: dict[str, list[str]] = {}
        for entity_name, task_ids in inverted_index.items():
            canonical = canonical_map.get(entity_name, entity_name)
            for tid in task_ids:
                if tid not in task_entities:
                    task_entities[tid] = []
                if canonical not in task_entities[tid]:
                    task_entities[tid].append(canonical)

        # 统计共现次数
        cooccurrence_count: dict[tuple[str, str], int] = {}
        cooccurrence_last_task: dict[tuple[str, str], str] = {}

        for tid, entities in task_entities.items():
            # 限制每个 task 内最多 15 个实体, 避免 C(n,2) 爆炸
            limited = entities[:15]
            if len(entities) > 15:
                logger.debug(
                    f"task {tid}: {len(entities)} entities > 15, truncated "
                    f"(dropped {len(entities) - 15})"
                )
            for e1, e2 in combinations(limited, 2):
                pair = tuple(sorted([e1, e2]))
                cooccurrence_count[pair] = cooccurrence_count.get(pair, 0) + 1
                cooccurrence_last_task[pair] = tid

        # 添加边
        for (e1, e2), count in cooccurrence_count.items():
            # 跳过自环
            if e1 == e2:
                continue
            G.add_edge(
                e1, e2,
                relation="co-occur",
                weight=min(count / 3.0, 1.0),  # 归一化: 3次共现 = 1.0
                source_task=cooccurrence_last_task.get((e1, e2), ""),
            )

        logger.info(
            f"共现图谱构建完成: {G.number_of_nodes()} 节点, "
            f"{G.number_of_edges()} 条共现边 "
            f"(来自 {len(task_entities)} 个 task 的共现关系)"
        )
        return G

    # --------------------------------------------------------
    # 保存 / 加载
    # --------------------------------------------------------

    def save_graph(self, G: nx.MultiDiGraph, gpickle_path: str, json_path: str, db_path: str = None, extraction_mode: str = "triple"):
        """保存图谱到 gpickle、JSON 和 SQLite（可选）"""
        # gpickle (networkx 3.x 移除了 write_gpickle, 用 pickle 替代)
        p = Path(gpickle_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(gpickle_path, "wb") as f:
            pickle.dump(G, f)
        logger.info(f"图谱已保存 (gpickle): {gpickle_path}")

        # JSON (node_link_data)
        data = nx.node_link_data(G, edges="links")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"图谱已保存 (JSON): {json_path}")

        # SQLite 持久化
        if db_path:
            from code_p5e_db import KGDatabase
            db = KGDatabase(db_path)
            db.import_graph(G, extraction_mode=extraction_mode)

    def save_entities(
        self,
        entity_map: dict[str, Entity],
        canonical_map: dict[str, str],
        output_file: str,
    ):
        """保存实体列表到 JSONL"""
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(p, "w", encoding="utf-8") as f:
            for name, entity in sorted(entity_map.items()):
                canonical = canonical_map.get(name, name)
                entity.canonical_name = canonical
                f.write(json.dumps(entity.to_dict(), ensure_ascii=False) + "\n")
                count += 1

        logger.info(f"已写入 {count} 个实体到 {output_file}")

    def save_inverted_index(
        self,
        inverted_index: dict[str, list[str]],
        canonical_map: dict[str, str],
        output_file: str,
    ):
        """
        保存倒排索引 (按 canonical name 聚合)

        输出格式:
        [
          {
            "canonical": "标准名",
            "aliases": ["别名1", "别名2"],
            "task_ids": ["task1", "task2"]
          }
        ]
        """
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)

        # 按 canonical name 聚合
        canonical_data: dict[str, dict] = {}
        for name, task_ids in inverted_index.items():
            canonical = canonical_map.get(name, name)
            if canonical not in canonical_data:
                canonical_data[canonical] = {"aliases": [], "task_ids": []}
            if name != canonical:
                canonical_data[canonical]["aliases"].append(name)
            for tid in task_ids:
                if tid not in canonical_data[canonical]["task_ids"]:
                    canonical_data[canonical]["task_ids"].append(tid)

        # 转换为列表并按 task 数降序排列
        result = []
        for canonical, data in sorted(
            canonical_data.items(),
            key=lambda x: len(x[1]["task_ids"]),
            reverse=True,
        ):
            result.append({
                "canonical": canonical,
                "aliases": data["aliases"],
                "task_count": len(data["task_ids"]),
                "task_ids": data["task_ids"],
            })

        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(
            f"倒排索引已保存: {output_file} "
            f"({len(result)} 个 canonical 实体)"
        )

    def save_stats(self, stats: KGStats):
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("Phase 5 知识图谱统计")
        logger.info(f"  抽取模式:         {stats.extraction_mode}")
        logger.info(f"  处理 task 数:     {stats.total_tasks}")
        if stats.extraction_mode == "triple":
            logger.info(f"  三元组总数:       {stats.total_triples}")
        logger.info(f"  实体总数 (原始):  {stats.total_entities}")
        logger.info(f"  唯一实体 (对齐后): {stats.unique_entities}")
        logger.info(f"  合并实体对:       {stats.merged_pairs}")
        logger.info(f"  图谱节点数:       {stats.total_nodes}")
        logger.info(f"  图谱边数:         {stats.total_edges}")
        if stats.extraction_mode == "entity":
            logger.info(f"  共现边数:         {stats.cooccurrence_edges}")
        logger.info("=" * 60)
