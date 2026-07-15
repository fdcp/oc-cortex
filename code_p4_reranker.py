"""
Phase 4 Reranker 模块
使用 Qwen3-Reranker-0.6B 做精排 (CausalLM 架构)

Qwen3-Reranker 是基于 CausalLM 的重排序模型:
  - 输入: system prompt + query + document
  - 输出: 对 "yes"/"no" token 的 logit 差异 → relevance score

正确用法: AutoModelForCausalLM + 计算 yes/no logits
"""
import os
import time
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class RerankResult:
    """单条 rerank 结果"""
    index: int          # 原始文档在输入列表中的索引
    score: float        # relevance score (0~1, softmax 后)
    document: str       # 原始文档文本 (可能截断)


class Qwen3Reranker:
    """
    Qwen3-Reranker-0.6B 精排器

    基于 CausalLM 架构，通过 yes/no token logit 差异计算相关性。
    """

    DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages."

    SYSTEM_PROMPT = (
        "Judge whether the Document matches the Intent. "
        "Your answer must be either `yes` or `no`."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = "cpu",
        cache_folder: Optional[str] = None,
        offline_mode: bool = True,
        instruction: Optional[str] = None,
        max_length: int = 8192,
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

        # 缓存目录
        if cache_folder is None:
            cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
        self.cache_folder = cache_folder
        self.offline_mode = offline_mode

        # 设备检测
        from code_p3_qdrant_store import _detect_device
        self.device = _detect_device(device)
        logger.info(f"Reranker 设备: {self.device}")

        # 加载模型
        import torch
        self._torch = torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"加载 Reranker 模型: {model_name}")
        t0 = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left", cache_dir=cache_folder
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=cache_folder
            )
        except Exception as e:
            if offline_mode and (
                "not found" in str(e).lower()
                or "no such file" in str(e).lower()
                or "offline" in str(e).lower()
            ):
                logger.warning("本地缓存未找到 Reranker 模型, 切换到在线模式下载")
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, padding_side="left", cache_dir=cache_folder
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_folder
                )
            else:
                raise

        # 确保 pad_token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 预计算 yes/no token IDs
        self._yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self._no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]
        logger.info(f"yes_token_id={self._yes_id}, no_token_id={self._no_id}")

        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Reranker 加载完成, 耗时 {time.time() - t0:.1f}s")

    def _build_prompt(self, query: str, document: str) -> str:
        """
        构造 Qwen3-Reranker 格式的 chat prompt

        格式:
        <|im_start|>system
        {SYSTEM_PROMPT}<|im_end|>
        <|im_start|>user
        <Instruct>: {instruction}
        <Query>: {query}
        <Document>: {document}<|im_end|>
        <|im_start|>assistant
        <think>

        """
        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"<think>\n\n</think>\n\n"
        )

    def rank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        对候选文档列表做精排

        Args:
            query: 用户查询
            documents: 候选文档文本列表
            top_k: 返回前 K 个结果

        Returns:
            按 score 降序排列的 RerankResult 列表
        """
        if not documents:
            return []

        torch = self._torch
        all_scores: list[float] = []

        t0 = time.time()
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            prompts = [self._build_prompt(query, doc) for doc in batch_docs]

            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # logits: [batch, seq_len, vocab_size]
                # 取最后一个有效 token 位置的 logits
                for b_idx in range(len(batch_docs)):
                    # 找到最后一个非 pad token 的位置
                    mask = attention_mask[b_idx]
                    last_pos = mask.sum().item() - 1
                    last_logits = outputs.logits[b_idx, last_pos, :]

                    # 取 yes 和 no 的 logit
                    yes_logit = last_logits[self._yes_id].item()
                    no_logit = last_logits[self._no_id].item()

                    # softmax over [yes, no]
                    exp_yes = torch.exp(torch.tensor(yes_logit))
                    exp_no = torch.exp(torch.tensor(no_logit))
                    score = exp_yes / (exp_yes + exp_no)
                    all_scores.append(score.item())

        elapsed = time.time() - t0
        logger.info(
            f"Reranker: {len(documents)} 篇文档, "
            f"{(len(documents) + self.batch_size - 1) // self.batch_size} batch, "
            f"耗时 {elapsed:.2f}s"
        )

        # 按 score 降序排序
        scored = [
            RerankResult(index=i, score=s, document=documents[i])
            for i, s in enumerate(all_scores)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)

        return scored[:top_k]
