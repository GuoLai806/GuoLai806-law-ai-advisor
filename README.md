# Evidence RAG

本项目提供通用、带版本和来源的证据检索服务，并保留可选法律资料 profile。默认入口是 `evidence_rag`；原仓库名称与历史代码保留，`legal_rag` 是调用同一核心的兼容层。

## 本机运行

使用 Python 3.12，在独立虚拟环境安装本项目：

```sh
python -m pip install .
python -m evidence_rag --help
```

可选 `.[dev]` 提供测试工具；`.[models]` 提供本地 embedding/reranker 依赖。权重不会随包安装或自动捆绑。`constraints-runtime.txt` 只是开发环境版本快照，尚未完成带哈希的跨平台锁和干净环境复验；正式部署前应由部署者固定并核验依赖。

通用 JSONL 每行至少有 `_id`、`title` 和 `text`；也可导入 UTF-8 文本、Markdown 或 PDF：

```sh
python -m evidence_rag ingest --db /tmp/evidence.sqlite3 --input /path/to/corpus.jsonl --corpus-id demo --revision r1
python -m evidence_rag activate --db /tmp/evidence.sqlite3 --revision r1
EVIDENCE_RAG_DB=/tmp/evidence.sqlite3 uvicorn evidence_rag.api:app --host 127.0.0.1 --port 8000
```

已有活动版本时，激活需以 `--expected <当前revision>` 做比较切换。默认不配置身份，业务请求会被拒绝；部署者需通过安全渠道提供 `EVIDENCE_RAG_AUTH`（token 到用户 ID 的 JSON 映射）。管理员导入还需 `EVIDENCE_RAG_ADMINS` 和 `EVIDENCE_RAG_INGEST_DIR`。静态 token 适合隔离开发，生产环境需单独完成认证、授权、TLS 和数据保护。默认公共语料对已认证用户共享；不要把私有文件放入共享索引。

`GET /health/live` 只检查进程；`GET /health/ready` 会检查活动索引和已配置组件。没有向量后端、重排器、生成模型或 tokenizer 时，系统可降级展示已检索资料，但不会把完整问答能力标为 ready。使用请求内 `budget_mode="conservative_bytes"` 是**需要服务端显式允许的试验选项**，不是精确 token 计数，也不是未知云端模板的形式化上界；默认仍要求匹配的 tokenizer。细节见[核心说明](evidence_rag/README.md)。

法律资料使用 `python -m legal_rag` 或 `uvicorn legal_rag.api:app`，历史 `main-system/server_text01.py` 现转到同一法律兼容应用，不再启动旧全局 Agent。法律语料仍需来源和版本核验；仓库 PDF 当前处于未审核状态。旧静态网页未适配新的鉴权与 SSE 协议，不应作为当前服务的验收入口。

## 可选容器配置

`main-system/Dockerfile` 使用**仓库根目录**作为 build context，运行通用 API，模型权重和凭据不打进镜像。`main-system/docker-compose.yml` 将端口绑定在本机 127.0.0.1 并使用独立数据卷。部署者可在明确配置好身份后执行 `docker compose -f main-system/docker-compose.yml up --build`；仓库中的配置本身没有启动 Docker，也没有验证目标主机上的容器。Docker healthcheck 仅检查进程存活，不等于 `/health/ready` 或检索质量通过。

详见[通用开发记录](docs/development/general-evidence-rag-20261106/WORK.md)、[技术选择](docs/development/general-evidence-rag-20261106/TECHNOLOGY-DECISIONS.md)和[法律资料记录](docs/development/legal-rag-20261105/WORK.md)。公开基准结果、人工审核状态与真实服务验证需分别查看对应证据，不从接口健康或合成样例推断准确率。

本轮模型对照、交付边界和执行步骤见 [用户版](docs/development/general-evidence-rag-20261106/PLAN-FOR-YOU.md) 与 [执行版](docs/development/general-evidence-rag-20261106/PLAN-FOR-AI.md)。高质量候选配置使用 `EVIDENCE_RAG_RERANKER=qwen`；BGE重排是较低延迟备选，一般生成的阈值仍须按目标语料另行校准。

本轮实测平台为macOS Apple Silicon/Python 3.12；受限文件读取和基准缓存锁使用POSIX接口，Windows需要额外适配。Linux容器只完成配置校验，尚无容器端到端验收。
