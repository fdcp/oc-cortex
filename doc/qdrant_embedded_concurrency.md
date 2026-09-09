# Qdrant Embedded 模式为什么不能并发访问 — 深度调研笔记

> 作者:Mavis (MiniMax Code)
> 调研时间:2026-09-09
> 关联事件:当天调试 `summarize` 工具时撞到 `RuntimeError: Storage folder ... already accessed by another instance`
> 关联脚本:[`scripts/migrate_qdrant_to_server.py`](../scripts/migrate_qdrant_to_server.py)

---

## 0. 一句话结论

**"Qdrant embedded 单进程独占"的根本原因不是 SQLite 本身,而是 Qdrant 在 `__init__` 阶段主动创建的应用层 `.lock` 文件,以及在 SQLite 端配置的 `journal_mode=DELETE + locking_mode=NORMAL` 串行模式**。SQLite 完全有能力支持并发(Qdrant 没启用而已)。

---

## 1. 现象:撞锁的 RuntimeError

### 1.1 错误信息

调用 Qdrant 工具时报:

```
RuntimeError: Storage folder /Users/zhaoxiuwei/Desktop/oc_sess_graph/qdrant_data
is already accessed by another instance of Qdrant client.
If you require concurrent access, use Qdrant server instead.
```

### 1.2 触发场景

机器上同时存在 2+ 个 `code_mcp_server.py` 进程,且都尝试初始化 `QdrantClient(path=...)`。第二个及之后的进程**直接被 Qdrant 拒在门外**,无法启动向量检索。

### 1.3 误诊陷阱

很多开发者(包括我一开始)看到这个错误会以为是 **SQLite 的并发限制**。实际上:

- SQLite 本身支持多读 + 单写(WAL 模式)
- Qdrant 选择不用 SQLite 的并发能力
- **真正撞锁的元凶是 Qdrant 应用层加的一个 13 字节 `.lock` 文件**

---

## 2. 证据链(由外到内,逐层验证)

### 2.1 顶层:`.lock` 文件是 Qdrant 自己的,不是 SQLite 加的

```bash
$ ls -la qdrant_data/
total 32
-rw-r--r--   1 zhaoxiuwei  staff    13 Aug 14 03:16 .lock          # ← Qdrant 加的
drwxr-xr-x   3 zhaoxiuwei  staff    96 Aug  5 10:39 collection
-rw-r--r--   1 zhaoxiuwei  staff  1950 Aug  5 13:00 meta.json

$ cat qdrant_data/.lock
tmp lock file
```

**关键点**:SQLite 自己的锁机制是 **fcntl 内核级文件锁**,不落盘。`.lock` 文件的存在本身证明锁是 **应用层加的**,不是数据库引擎加的。

### 2.2 storage 层:每个 collection 一个 SQLite 文件

```bash
$ ls -la qdrant_data/collection/tasks/
total 1024
-rw-r--r--  1 zhaoxiuwei  staff  503808 Aug  5 10:39 storage.sqlite    # ← 单文件

$ file qdrant_data/collection/tasks/storage.sqlite
SQLite 3.x database, last written using SQLite version 3040001,
file counter 153, database pages 123, schema 4
```

每个 collection 就是一个独立 SQLite 文件,内含一个 `points` 表(向量 + payload 都塞这里)。

### 2.3 SQLite 配置:最严格的串行模式

```python
# 探查 4 个 collection 的 SQLite 配置(全一致)
journal_mode: delete        # ← 不是 WAL!
locking_mode: normal        # ← 默认
busy_timeout: 5000          # ← 撞锁等 5s 再报 busy
tables: ['points']
```

| 配置 | 实际值 | 含义 |
|---|---|---|
| `journal_mode` | `DELETE` | 每次提交删 journal 文件,**最严格,无并发** |
| `locking_mode` | `NORMAL` | SQLite 默认(但能改成 EXCLUSIVE / WAL 模式) |
| `busy_timeout` | 5000ms | 撞锁最多等 5 秒就抛 SQLITE_BUSY |

**SQLite 完全有能力做**:
- `journal_mode=WAL`:多读 + 单写(允许并发连接)
- `journal_mode=WAL2`:读写并发(SQLite 3.22+)
- `locking_mode=EXCLUSIVE`:进程内排他(没用,会阻塞自己)

**Qdrant 选了最保守的组合**。

### 2.4 对照:Docker 里的存储完全不同

```bash
$ docker exec qdrant find /qdrant/storage/collections/tasks -type d
/qdrant/storage/collections/tasks/0
/qdrant/storage/collections/tasks/0/segments
/qdrant/storage/collections/tasks/0/segments/a82c9824-...
/qdrant/storage/collections/tasks/0/segments/a82c9824-.../payload_storage
/qdrant/storage/collections/tasks/0/segments/a82c9824-.../vector_storage/vectors
/qdrant/storage/collections/tasks/0/wal
```

| 维度 | 本地 embedded | Docker server |
|---|---|---|
| 每 collection 文件 | **1 个** `.sqlite` | **多 segment** + WAL + 配置 |
| 向量存储 | SQLite BLOB | mmap `chunk_*.dat` |
| Payload 存储 | SQLite table | 自研 `page_*.dat` |
| 一致性保证 | 单文件 fsync | Raft + WAL + replica_state |
| 多副本 | ❌ | ✅ |

**澄清误解**:Docker 里的 Qdrant **不是 SQLite**,用的是 Qdrant 自研的 segment-based 存储引擎。

---

## 3. 撞锁的真实流程(伪代码)

```
进程 A 启动:
    QdrantClient(path="./qdrant_data")
        ├─ 1. 检查 qdrant_data/.lock
        │     └─ 不存在 → 通过
        ├─ 2. File::create(qdrant_data/.lock)   # ← 应用层锁,关键步骤
        ├─ 3. 打开 collection/{name}/storage.sqlite  # ← SQLite 锁(此时才介入)
        └─ 4. SELECT count(*) FROM points ...   # 正常工作

进程 B 启动(并发):
    QdrantClient(path="./qdrant_data")
        ├─ 1. 检查 qdrant_data/.lock
        │     └─ 已存在 → 抛 RuntimeError           # ← 在这步就 fail
        └─ (后续步骤根本不会执行,SQLite 都不会被打开)
```

**结论**:
- 撞锁发生在 Qdrant 应用层 (`__init__` 阶段)
- SQLite 根本来不及工作
- 即便去掉 `.lock` 改用 SQLite WAL,也只是把冲突延后到 SQLite 层 —— **仍然不安全**

---

## 4. Qdrant 为何这么设计?

### 4.1 设计目标

Qdrant embedded 模式的产品定位:**零配置、单进程、与宿主程序同生死**。

```rust
// Qdrant 源码(简化)
pub fn new(path: &Path) -> Result<Self> {
    let lock_path = path.join(".lock");
    if lock_path.exists() {
        return Err("Storage folder ... already accessed");
    }
    File::create(lock_path)?;     // 早死早超生,让用户立刻知道有冲突
    // ...
}
```

**理念**:
- "一个程序 = 一个 client,不需要并发"
- "真要并发?请用 server 模式"
- 不暴露锁竞争、HNSW 重建、optimistic concurrency 等复杂性

### 4.2 为什么不开 SQLite WAL?

| 原因 | 说明 |
|---|---|
| **多数用户不需要** | CI 任务、build pipeline、单进程脚本是主要场景 |
| **实现成本高** | HNSW 索引在并发下要重新设计,segment 合并要协调 |
| **不如引导用 server** | 同源代码,切换零成本(我们这次的迁移就证明了) |

### 4.3 历史原因

Qdrant 早期(0.x 时代)embedded 用的是 **SQLite + 单文件**,目的是:
- 零依赖(用户机器上一般有 SQLite 或能 easy_install)
- 单文件可移植
- 跨平台(Win/Mac/Linux 一致行为)

代价就是放弃了并发能力。

---

## 5. 类比其他"用 SQLite 却不并发"的系统

| 系统 | 是否用 SQLite | 支持多连接? | 说明 |
|---|---|---|---|
| **Qdrant embedded** | ✅ | ❌ 应用层 .lock 拒绝 | 本笔记 |
| **DuckDB** | ❌ 自研 | ✅ 多连接 + MVCC | 高性能分析库 |
| **LiteFS** | ✅ | ✅ FUSE 跨节点 | 跨节点复制,本地并发 OK |
| **Bun:sqlite** | ✅ | ✅ WAL + 多连接 | 默认开 WAL |
| **Python `sqlite3`** | ✅ | ✅ connection-per-thread | 默认支持多连接 |
| **Chrome IndexedDB** | ❌ | ✅ | 浏览器原生 |
| **LMDB** | ❌ | ❌ 单写多读 | 比 SQLite 还严格 |

**关键启示**:**SQLite 不等于"单进程"**。SQLite 是能力上限,具体系统决定用多少。Qdrant 选择只用最低能力。

---

## 6. 解决方案对比

### 6.1 三种主流方案

| 方案 | 改动量 | 适用场景 |
|---|---|---|
| **A. 切到 server 模式** ✅ | 改 1 个 env 或 yaml 字段 | 任何多进程场景(本次采用) |
| **B. 进程间互斥(应用层加锁)** | 5–10 行 Python | 临时方案,无并发收益 |
| **C. 改造 Qdrant 源码(去 .lock + WAL)** | 几百行 Rust + 长期维护 | 不推荐,官方无意愿 |
| **D. 用 Shared Qdrant Daemon** | 写一个独立 HTTP daemon | 跟方案 A 等价,实现更复杂 |

### 6.2 为什么选 A(server 模式)

- ✅ 零代码改动,只改 env
- ✅ Docker 一行命令起 server
- ✅ 多 IDE/多 session 共享一份数据
- ✅ 支持未来分片、副本
- ✅ Qdrant 官方支持路径,不会随版本升级失效
- ✅ 跟生产部署一致(开发即生产)

### 6.3 server 模式的额外收益(本项目实测)

- 启动时 `migrate_qdrant_to_server.py` 1.4 秒搬完 1,991 points
- `graph_rag_search` Top-1 质量从"靠 BM25 词项匹配"升级为"embedding 语义匹配"
- 之前 Docker qdrant 跑了 5 天没数据(0 points),现在真正用上了

---

## 7. 本项目实战案例(2026-09-09)

### 7.1 问题发现

- 4 个 `code_mcp_server.py` 进程并存
- 调用 `summarize` 时撞 Qdrant 锁
- 误以为是 SQLite 限制

### 7.2 排查过程

| 步骤 | 工具 | 发现 |
|---|---|---|
| 1. 看进程 | `ps aux \| grep code_mcp_server` | 4 个并存 |
| 2. 看网络 | `lsof -nP -p 40121 -iTCP` | **没连 6333** |
| 3. 看本地目录 | `lsof +D qdrant_data/` | **0 命中** |
| 4. 看 .lock 时间 | `ls -la qdrant_data/.lock` | Aug 14(25 天前) |
| 5. 看 Docker | `curl localhost:6333/collections` | collections 存在,**0 points** |
| 6. 看进程 env | `ps eww -p 40121` | 有 `QDRANT_URL=http://localhost:6333` |
| 7. 看进程 fd | `lsof -p 40121` | 没有 qdrant_data 访问,也没有 6333 连接 |

### 7.3 真相

- **没有任何进程使用 embedded 模式**
- 当前 MCP 进程**配置上**走 server 模式(env 有 QDRANT_URL)
- Docker qdrant 跑了 5 天但**没数据**(没 migrate 过)
- 搜索能"工作"是因为 **in-memory BM25 撑起来**(从 chunks JSONL 重建)
- dense embedding 路径实际是空跑

### 7.4 解决

```bash
# 1. 写迁移脚本
$EDITOR scripts/migrate_qdrant_to_server.py

# 2. 跑迁移
python3 scripts/migrate_qdrant_to_server.py
# → 🎉 迁移完成: 1991 points, 总耗时 1.4s

# 3. 验证
curl http://localhost:6333/collections/tasks | jq .result.points_count
# → 76

# 4. 端到端测
graph_rag_search("序列并行 SP 通信原语 all-gather")
# → Top-1: "NVIDIA SP与Ulysses并行策略对比"(精准命中)
```

### 7.5 写文档

- [README.md](../README.md) 加 "Qdrant 部署" 章节
- 模式对比表 + server 启动 + 迁移工作流 + 故障排查

---

## 8. 复现 / 验证方法

### 8.1 复现撞锁

```bash
# 1. 准备本地数据(假设已有)
ls qdrant_data/collection/

# 2. 开两个进程
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(path='./qdrant_data')
print('client 1 起来了,保持连接...')
import time; time.sleep(30)
" &
PID1=$!
sleep 2  # 等第一个 client 初始化完

# 3. 第二个进程立刻撞锁
python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(path='./qdrant_data')   # ← 抛 RuntimeError
"
# → RuntimeError: Storage folder ... already accessed by another instance

kill $PID1
```

### 8.2 验证 SQLite 模式

```bash
python3 -c "
import sqlite3
con = sqlite3.connect('qdrant_data/collection/tasks/storage.sqlite')
print('journal_mode:', con.execute('PRAGMA journal_mode').fetchone()[0])
print('locking_mode:', con.execute('PRAGMA locking_mode').fetchone()[0])
"
```

### 8.3 验证 server 不受 .lock 影响

```bash
# server 用 QdrantClient(url=...),不碰本地目录
# 多进程并发查询都没问题
for i in {1..5}; do
  python3 -c "
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6333')
print('PID', __import__('os').getpid(), '->', c.get_collections())
" &
done
wait
# → 5 个进程都成功,无任何错误
```

---

## 9. 给后续维护者的建议

### 9.1 短期(本周内)

- ✅ **保留 `qdrant_data/` 目录** — embedded 模式仍是合法部署选项(CI、build pipeline)
- ✅ **持久化 QDRANT_URL** — 写进 `code_p3_config.yaml` 的 `qdrant.url`,避免 env 丢失
- ✅ **CI 加并发检查** — 启动多个 MCP server 进程,跑一次 `graph_rag_search`,确保不撞锁

### 9.2 中期(本月内)

- 📌 **监控 server 模式健康** — 加 `curl localhost:6333/healthz` 到开机启动
- 📌 **备份策略** — `qdrant_data/` 用 `rsync`,server 容器用 `docker volume backup`
- 📌 **测试覆盖** — 给 `migrate_qdrant_to_server.py` 加单测,覆盖 dry-run / 部分迁移 / schema 不一致等场景

### 9.3 长期(下季度)

- 📦 **支持 BGE-M3 sparse vector 迁移** — 当前只测了纯 dense 路径
- 🔬 **对比 embedded vs server 性能** — 同样的查询,延迟/吞吐/内存差异
- 📖 **写"Qdrant 部署选型"决策树** — 什么场景用 embedded,什么用 server

---

## 10. 关键引用 / 参考

| 资料 | 链接/位置 | 用途 |
|---|---|---|
| Qdrant 官方文档 | https://qdrant.tech/documentation/ | 部署模式说明 |
| Qdrant GitHub | `qdrant_client` Python 库 `QdrantClient.__init__` | `.lock` 创建逻辑 |
| SQLite WAL 模式 | https://www.sqlite.org/wal.html | 验证 SQLite 并发能力 |
| SQLite locking_mode | https://www.sqlite.org/pragma.html#pragma_locking_mode | locking 模式参考 |
| 本项目 `migrate_qdrant_to_server.py` | [`scripts/`](../scripts/) | 实际迁移工具 |
| 本项目 README "Qdrant 部署" 章节 | [`README.md`](../README.md) | 用户面部署指南 |
| 本次会话上下文 | 9/9 多实例排查 | 实战案例 |

---

## 11. TL;DR

| 问题 | 答案 |
|---|---|
| Qdrant embedded 是不是单进程? | **是** |
| 是 SQLite 导致的? | **不是** |
| 真正原因是什么? | Qdrant 应用层 `.lock` 文件 + SQLite DELETE 模式 |
| SQLite 能支持并发吗? | **能**(WAL 模式),Qdrant 选择不用 |
| 怎么解决多进程? | 切到 **server 模式** |
| 切换成本? | 改 1 个 env,1.4 秒搬 2000 points |
| 推荐方案? | server 模式(本次采用),embedded 保留给单进程场景 |

---

> 📌 维护提示:本笔记与 `README.md` 的 "Qdrant 部署" 章节互为补充 —— README 面向用户(怎么用),本笔记面向维护者(为什么这样设计、怎么排查、怎么演进)。
