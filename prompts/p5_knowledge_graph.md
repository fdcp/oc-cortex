# Phase 5 Prompts: Knowledge Graph Construction

---

## 1. Triple Extraction (`TRIPLE_EXTRACTION_PROMPT`)

**Source**: `src/code_p5_kg_builder.py`

```
你是一个跨 session 知识图谱构建助手。请从以下"任务摘要"中抽取 (实体, 关系, 实体) 三元组。

【核心原则】
这些三元组将用于跨 session 知识关联,因此:
- 实体必须是**可能被其他 session 也提及的**概念/项目/模块/文件/工具/技术
- 实体不应该是一次性的属性值、公式、ID、或仅在该 session 内出现的临时术语

【实体要求】
合格的实体类型:
- 项目/仓库名 (如 opencode, transformers)
- 模块/组件名 (如 认证模块, 消息队列)
- 文件名 (如 config.py, package.json)
- 技术概念/框架 (如 RoPE位置编码, BM25, PyTorch)
- 工具/库 (如 Qdrant, jieba, NetworkX)
- Bug/问题描述 (如 token过期bug, 内存泄漏问题)

不合格的实体(必须排除):
- 属性值: [seq_len, dim//2], 0.95, v3.2
- 数学公式: m·θ_i, e^(iθ)
- Session/会话 ID: ses_0a53b68...
- 过于通用的词: 用户, 代码, 系统, 方法, 问题

【关系要求】
使用精确的动词短语,优先选择:
  实现, 调用, 依赖, 修复了, 配置了, 属于, 使用, 修改了,
  优化了, 集成了, 替代了, 对比了, 新增了, 删除了, 迁移了

禁止使用的关系: 是, 分为, 包含(除非是严格的模块-子模块关系), 形状为, 元素表达式

【输出格式】严格 JSON, 输出 3-8 个高质量三元组:
{{
  "triples": [
    {{"head": "实体1", "relation": "精确关系", "tail": "实体2", "confidence": 0.9}}
  ]
}}

重要: 你必须直接在 content 中输出上述 JSON 对象。不要把 JSON 放在 reasoning/思考过程中。
如果你使用了 reasoning/思考过程, 请在思考结束后, 将完整的 JSON 对象作为你的最终回答输出。

好的例子:
  ({{"head": "LLaMA", "relation": "使用", "tail": "RoPE位置编码", "confidence": 0.95}})
  ({{"head": "opencode", "relation": "依赖", "tail": "SQLite", "confidence": 0.9}})
  ({{"head": "RoPE位置编码", "relation": "实现于", "tail": "precompute_freqs_cis函数", "confidence": 0.85}})

坏的例子:
  ({{"head": "freqs张量", "relation": "形状为", "tail": "[seq_len, dim//2]"}}) ← 属性值不是实体
  ({{"head": "消息模型", "relation": "包含", "tail": "agent"}}) ← "包含"太笼统
  ({{"head": "最新会话ID", "relation": "是", "tail": "ses_0a53b68..."}}) ← ID不是实体

【任务摘要】
{task_summary}
```

---

## 2. Entity Merge (Single Pair) (`ENTITY_MERGE_PROMPT`)

**Source**: `src/code_p5_kg_builder.py`

```
判断以下两个实体在给定的上下文中是否应该合并为同一个概念。

实体 A: {entity_a}
上下文 A: {context_a}

实体 B: {entity_b}
上下文 B: {context_b}

合并规则:
- 完全同义或互为别名: 合并
- 含义不同或属于不同层级/上下文: 不合并
仅回答 "MERGE" 或 "KEEP"。
```

---

## 3. Batch Entity Merge (`BATCH_ENTITY_MERGE_PROMPT`)

**Source**: `src/code_p5_kg_builder.py`

```
你是一个知识图谱实体对齐助手。请判断以下实体对是否应该合并为同一个概念。

【合并规则】
- 完全同义或互为别名: MERGE
- 含义不同或属于不同层级/上下文: KEEP

【待判断实体对】
{pairs_text}

【输出格式】
严格返回JSON数组，每个元素是包含"pair"和"action"的对象，不要返回扁平列表，不要添加任何额外文本：
[
  {{"pair": ["实体A", "实体B"], "action": "MERGE"}},
  {{"pair": ["实体C", "实体D"], "action": "KEEP"}}
]

【示例】
输入: ["OpenCode", "opencode", "SQLite", "数据库"]
输出:
[
  {{"pair": ["OpenCode", "opencode"], "action": "MERGE"}},
  {{"pair": ["SQLite", "数据库"], "action": "KEEP"}}
]

重要: 不要把 JSON 放在 reasoning/思考过程中, 必须直接在 content 中输出 JSON。每对必须有一个判断。
```

---

## 4. Entity Extraction (`ENTITY_EXTRACTION_PROMPT`)

**Source**: `src/code_p5_kg_builder.py`

```
你是一个跨 session 知识关联助手。请从以下"任务摘要"中提取 5-10 个关键实体。

【合格的实体类型】
- 项目/仓库名 (如 opencode, transformers)
- 模块/组件名 (如 认证模块, 消息队列)
- 文件名 (如 config.py, package.json)
- 技术概念/框架 (如 RoPE位置编码, BM25, PyTorch)
- 工具/库 (如 Qdrant, jieba, NetworkX)
- Bug/问题描述 (如 token过期bug, 内存泄漏问题)

【排除】
- 一次性属性值: [seq_len, dim//2], 0.95, v3.2
- 数学公式: m·θ_i, e^(iθ)
- Session/会话 ID: ses_0a53b68...
- 过于通用的词: 用户, 代码, 系统, 方法, 问题

【输出格式】严格 JSON:
{{"entities": ["实体1", "实体2", "..."]}}

输出 5-10 个高质量实体, 不要添加解释。

重要: 你必须直接在 content 中输出上述 JSON 对象。不要把 JSON 放在 reasoning/思考过程中。

【任务摘要】
{task_summary}
```
