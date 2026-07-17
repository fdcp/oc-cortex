## Phase 6b: 技术决策溯源 + 主题总结骨架

### 概述

在 Phase 6 跨 Session 总结基础上增加两项能力：

1. **技术决策溯源** — 从 SQLite 知识图谱中按关系类别多跳追踪决策链，可回答"为什么选 Qdrant""A 和 B 的选型对比"
2. **主题总结骨架** — 从图谱中定位主题节点 → BFS 扩散获取相关任务 → 注入决策链上下文 → LLM 生成带因果关系的结构化总结

### 新增文件

| 文件 | 说明 |
|------|------|
| `code_p6b_skeleton.py` | 核心模块：`DecisionTracer`（决策溯源）+ `SummarySkeleton`（骨架构建 + LLM 生成） |
| `code_p6b_cli.py` | CLI 入口：`trace`（决策溯源）、`summary`（骨架总结）、`search`（实体搜索） |
| `code_p6b_config.yaml` | Phase 6b 配置：决策关系分类、BFS 参数、数据库路径 |

### 依赖

| 模块 | 用途 |
|------|------|
| `code_p5e_db` | `KGDatabase` — SQLite 知识图谱查询 |
| `code_p6_summarizer` | `SessionSummarizer` — Phase 6 总结器（复用搜索和 LLM 客户端） |
| `code_p1_utils` | `count_tokens`、`safe_truncate` |

### 决策关系分类（宽泛匹配）

| 类别 | 关系关键词 | 说明 |
|------|-----------|------|
| 选用 | 使用, 集成了, 实现, 支持, 通过兼容层加载 | 技术采纳决策 |
| 选型 | 对比, 对比了, 区别于, 适用于, 适合 | 技术选型比较 |
| 排除 | 禁用, 缺少, 阻塞于, 阉割了 | 技术排除/限制 |
| 替换 | 替代了, 新增了, 补充了 | 技术迭代替换 |
| 依赖 | 属于, 包含, 对应 | 层级依赖关系 |

---

### 操作步骤（可复现）

#### 0. 前置准备

```bash
# 设置 API Key
export OPENCODE_ZEN_API_KEY=$(python3 -c \
  "import json; d=json.load(open('$HOME/.local/share/opencode/auth.json')); print(d['opencode-go']['key'])")

# 离线模式（避免 HuggingFace 超时）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 进入项目目录
cd ~/Desktop/oc_sess_graph

# 确认前置数据就绪
ls output/triple/knowledge_graph.db  # Phase 5e 产物
```

#### 1. 决策溯源

```bash
# 基本溯源
python3 code_p6b_cli.py trace "FlashAttention"

# 指定跳数 + JSON 输出
python3 code_p6b_cli.py trace "OpenCode" --hops 2 --json

# 搜索实体（确认实体存在）
python3 code_p6b_cli.py search "Flash"
```

**预期输出**（FlashAttention）：

```
## 决策链 [FlashAttention] (4 步, 2 跳)

### 依赖类
  - FlashAttention → Compute-Bound (属于, 1跳) [ses_0c5b6171...]

### 选型类
  - FlashAttention ← PyTorch SDPA (对比了, 1跳) [ses_0c5b6171...]

共 4 个决策步骤, 4 个关联实体, 最大 2 跳
```

#### 2. 骨架主题总结

```bash
# 基本骨架总结（含决策链注入）
python3 code_p6b_cli.py summary "FlashAttention"

# 不注入决策链（仅用骨架结构）
python3 code_p6b_cli.py summary "FlashAttention" --no-trace

# 指定参数
python3 code_p6b_cli.py summary "OpenCode" --max-tasks 5 --hops 2 --json
```

**预期输出**：LLM 生成的结构化总结，包含：
- 决策逻辑主线（为什么选择、如何对比）
- 技术细节与关键结论
- task_id 引用追溯
- 整体评价与后续建议

#### 3. Python API

```python
from code_p5e_db import KGDatabase
from code_p6_summarizer import SessionSummarizer
from code_p6b_skeleton import DecisionTracer, SummarySkeleton

# 初始化
db = KGDatabase("output/triple/knowledge_graph.db")
summarizer = SessionSummarizer("code_p3_config.yaml")
tracer = DecisionTracer(db, max_hops=3)
skeleton = SummarySkeleton(db, summarizer, tracer)

# 决策溯源
chain = tracer.trace_decision("FlashAttention")
print(tracer.format_context(chain))

# 收集决策链关联的 task_id
task_ids = tracer.get_related_tasks(chain)

# 骨架总结
result = skeleton.generate_summary(
    topic="FlashAttention 的实现和优化",
    use_decision_trace=True,
    max_tasks=5,
)
print(result.summary)
print(f"决策链: {len(result.decision_chain.steps)} 步, {result.decision_chain.hop_count} 跳")
print(f"耗时: {result.elapsed_ms}ms")
```

---

### 端到端测试结果

#### 决策溯源

| 实体 | 决策步骤 | 关联实体 | 跳数 | 耗时 |
|------|---------|---------|------|------|
| FlashAttention | 4 | 4 | 2 | <1ms |
| OpenCode | 34 | 38 | 2 | <1ms |
| 序列并行 | 0 | 5 | 0 | <1ms |

注：序列并行的边关系（'为', '切分', '正交于'）不在决策类别中，因此决策步骤为 0。

#### 骨架总结

| 指标 | 值 |
|------|------|
| 主题 | FlashAttention |
| 决策步骤 | 4 步 |
| 关联任务 | 1 个 |
| 总耗时 | 17.6s（含模型加载 + LLM） |
| 输出质量 | 结构化总结含决策主线、选型对比、技术细节、建议 |

与 Phase 6 纯向量总结对比，骨架总结的输出多了**决策因果链**和**选型对比**维度，LLM 按"决策逻辑 → 选型对比 → 技术细节 → 评价建议"组织内容。

### 架构

```
DecisionTracer.trace_decision(entity)
  │
  ├─ _resolve_entity()   精确匹配 → 模糊匹配
  │
  ├─ BFS 遍历 edges 表
  │    ├─ 查询出边 + 入边
  │    ├─ 匹配决策类别 (DECISION_CATEGORIES)
  │    └─ visited 防环 + max_visited 防爆
  │
  └─ 去重 + 排序 → DecisionChain

SummarySkeleton.generate_summary(topic)
  │
  ├─ Stage 1: _resolve_topic()   定位主题实体
  │    └─ 降级: 纯向量总结
  │
  ├─ Stage 2: BFS 扩散           获取相关实体
  │
  ├─ Stage 3: 收集关联 Task      按时间排序
  │
  ├─ Stage 4: 决策链追踪         DecisionTracer
  │
  ├─ Stage 5: _build_sections()  构建结构化大纲
  │
  └─ Stage 6: LLM 生成           骨架 + 决策链 + Task → 结构化总结
```
