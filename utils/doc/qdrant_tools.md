# Utils 工具文档

> 工程级运维脚本(数据迁移 / 备份恢复)的用法与最佳实践。
> 两个脚本都直接走 Qdrant HTTP API,不依赖 Python `qdrant-client` 库。

## 工具总览

| 脚本 | 用途 | 触发场景 |
|---|---|---|
| `migrate_qdrant_to_server.py` | embedded → server **一次性**数据迁移 | 项目从单机 embedded 切到多实例共享 server |
| `qdrant_snapshot.py` | collection 快照管理(backup / restore / list / download) | 灾备、版本升级、跨实例复制 |

---

## migrate_qdrant_to_server.py

### 何时用

- 项目从 embedded Qdrant(SQLite 后端)切到 server(RocksDB 后端)
- `qdrant_data/` 已有数据,需搬到 Docker Qdrant 容器
- **不是**常规备份工具 —— 迁完即弃,后续用 snapshot 工作流

### 用法

```bash
# 1. 预览(不写数据)
python3 utils/migrate_qdrant_to_server.py --dry-run

# 2. 全量迁移(默认 ./qdrant_data → http://localhost:6333)
python3 utils/migrate_qdrant_to_server.py

# 3. 子集迁移
python3 utils/migrate_qdrant_to_server.py --collections tasks chunks_summary
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--src` | `./qdrant_data` | embedded 数据目录 |
| `--dst` | `http://localhost:6333` | 目标 Qdrant URL |
| `--collections` | 全部 | 只迁指定 collection |
| `--dry-run` | False | 只打印 schema |
| `--batch-size` | 200 | scroll 批次 |
| `--no-recreate` | False | 追加模式(dst collection 必须存在) |

### 限制

- embedded 与 server 存储格式不同,**不能**用 `-v` 互挂
- 迁移走 HTTP upsert,**会**丢掉原始 HNSW 索引,server 端重建耗时 = collection 大小
- 不跨大版本(Qdrant v1.19 迁到 v1.18 不保证可读)

---

## qdrant_snapshot.py

### 何时用

- 容器销毁前的**紧急备份**
- 升级 Qdrant 版本前快照
- 跨实例复制(开发 → 生产)
- 把生产 snapshot 拉到 staging 调试

### 用法

```bash
# 1. 备份全部 collections
python3 utils/qdrant_snapshot.py backup --out ./qdrant_snapshots/20260910

# 2. 备份子集
python3 utils/qdrant_snapshot.py backup --out ./backups --collections tasks entities

# 3. 列 server 上现有 snapshots
python3 utils/qdrant_snapshot.py list
python3 utils/qdrant_snapshot.py list --collection tasks

# 4. 从 host 恢复(容器重建场景)
python3 utils/qdrant_snapshot.py restore --in ./qdrant_snapshots/20260910
python3 utils/qdrant_snapshot.py restore --in ./backups --collection tasks

# 5. 下载单个 snapshot
python3 utils/qdrant_snapshot.py download \
  --collection tasks \
  --name tasks-2026-09-10-12-34-56.snapshot \
  --out ./backups/
```

### 参数

| 子命令 | 参数 | 说明 |
|---|---|---|
| 全局 | `--host` | Qdrant URL(默认 `http://localhost:6333`) |
| `backup` | `--out` | host 输出目录 |
| `backup` | `--collections` | 子集,默认全部 |
| `restore` | `--in` | snapshot 目录 |
| `restore` | `--collection` | 只恢复指定 collection |
| `restore` | `--wait` / `--no-wait` | 是否等 collection ready(默认 wait) |
| `list` | `--collection` | 只列指定 collection |
| `download` | `--collection` `--name` `--out` | 精确下载单个 |

### 文件命名约定

下载后 snapshot 文件统一命名:

```
{collection}__{original_snapshot_name}.snapshot
```

例:`tasks__tasks-2026-09-10-12-34-56.snapshot`

`restore` 按此反解析 collection,所以**别手动改 snapshot 文件名**。

### snapshot 文件格式

Qdrant 私有**二进制**(不是 JSON),扩展名 `.snapshot`,包含:
- 全部向量
- 全部 payload
- HNSW 索引
- collection config

不能用 `cat` / `grep` 看内容。跨大版本**不保证兼容**。

---

## 已知陷阱

### 1. HTTP 代理拦截 localhost

如果 shell 设了 `http_proxy` / `https_proxy` / **`all_proxy`(SOCKS5)**,`requests` 和 `curl` 会自动走代理,localhost 请求会:

```
requests.exceptions.HTTPError: 502 Server Error: Bad Gateway
# 或
curl: Recv failure: Connection reset by peer (SOCKS5)
```

**完整 unproxy 列表**(5 个变量,大小写都算):

```bash
unset all_proxy all_PROXY ALL_PROXY \
      http_proxy HTTP_PROXY \
      https_proxy HTTPS_PROXY
```

**解决方案**:

| 脚本 | 怎么处理 |
|---|---|
| `qdrant_snapshot.py` | ✅ 内部用 `_NO_PROXY_SESSION.trust_env = False` 已 bypass (连 `all_proxy` 一起绕开) |
| `migrate_qdrant_to_server.py` | ⚠️ **未处理**,跑前需手动执行上面 5 个 `unset` |

### 2. 容器无 `-v` 挂载时 snapshots 必须留 host 一份

Qdrant container 若启动没加 `-v /host/path:/qdrant/storage`,容器销毁 → 数据 + 容器内 snapshots 全丢。

**必须做**:`backup` 时把 snapshot **下载到 host**(脚本默认就这么做),否则容器一旦死就是零。

### 3. snapshot 不跨 Qdrant 大版本

v1.19 生成的 snapshot 不能在 v1.18 恢复。跨大版本升级前**先在测试环境验证**。

### 4. `restore` 会覆盖现有 collection

`PUT /collections/{name}/snapshots/recover?priority=snapshot` 会用 snapshot 覆盖同名 collection,**不会合并**。

---

## 推荐工作流

### 日常(防 crash)

```bash
# 1. 容器 --restart=always 保活
docker run -d --name qdrant --restart=always \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant:v1.19.0

# 2. 定期 snapshot backup (cron / 手动)
python3 utils/qdrant_snapshot.py backup --out /backup/qdrant/$(date +%Y%m%d)
```

### 容器重建(数据迁到新机器)

```bash
# 旧机器
python3 utils/qdrant_snapshot.py backup --out ./pre_migration/
scp -r ./pre_migration/ newhost:/tmp/

# 新机器
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v /host/qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.19.0
python3 utils/qdrant_snapshot.py restore --in /tmp/pre_migration/
```

### embedded → server(老项目一次切换)

```bash
# 1. 启动空 server 容器
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.0

# 2. 一次性迁移(只能跑一次)
python3 utils/migrate_qdrant_to_server.py --dry-run
python3 utils/migrate_qdrant_to_server.py

# 3. 之后走 snapshot 工作流
python3 utils/qdrant_snapshot.py backup --out ./first_backup/
```

---

## 关联文档

- `../../doc/qdrant_embedded_concurrency.md` — embedded 为什么不能并发(理解为什么要切到 server)
- `../../doc/code_p3_README.md` — Phase 3 总体流程
- `../../README.md` "Qdrant 部署" 章节 — 用户面部署策略