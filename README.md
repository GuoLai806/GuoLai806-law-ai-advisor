# Evidence RAG

本项目提供通用、带版本和来源的证据检索服务，并保留可选法律资料 profile。默认入口是 `evidence_rag`；原仓库名称与历史代码保留，`legal_rag` 是调用同一核心的兼容层。

真实 OpenViking 全库用户流程评测见 [大白话说明](docs/development/openviking-user-flow-20261106/PLAN-FOR-YOU.md)、[执行依据](docs/development/openviking-user-flow-20261106/PLAN-FOR-AI.md) 和 [复现步骤](docs/development/openviking-user-flow-20261106/REPRODUCTION.md)。服务根路径 `/` 已有新问答网页，使用真实 SSE、引用原文展开和停止撤回。下方历史参考检索分数没有经过真实 OpenViking，不能替代这一轮结果。

本轮真实OpenViking全流程已完成：SciFact 5,183篇、300题，检索nDCG@10=0.7942、Recall@20=93.59%；标签一致212/300、引用文档标签F1=.5358、完整流程269/300。与同题BM25实际排名相比nDCG相对提升18.73%，仅指检索阶段。全部失败保留，正式运行跨主机睡眠，逐题秒表不代表真实日历等待或生产延迟，详见[结果及限制](docs/development/openviking-user-flow-20261106/RESULTS.md)。

30题质量优化已完成真实OpenViking同题对照：旧21/30、新22/30，引用文档标签F1 .5614→.7143，流程27→29/30；小样本总体差区间跨0、等待增加，默认仍text。见[大白话说明](docs/development/openviking-quality-20261107/PLAN-FOR-YOU.md)、[执行依据](docs/development/openviking-quality-20261107/PLAN-FOR-AI.md)、[完整结果](docs/development/openviking-quality-20261107/RESULTS.md)、[选型](docs/development/openviking-quality-20261107/TECHNOLOGY-DECISIONS.md)与[简历建议](docs/development/openviking-quality-20261107/RESUME-REWRITE.md)。

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

可复现的本地BGE嵌入端点与OpenViking0.5.3配置见 [examples](examples/README.md)。模型环境版本快照为 `constraints-models-tested.txt`，仍需固定公开模型revision并在仓库外下载权重。

## 历史参考检索测试：没有经过真实 OpenViking

BEIR SciFact完整5,183文档、官方300查询：预选BGE-M3+Qwen3-Reranker-4B的候选排序nDCG@10 **0.794175**、Recall@20 **0.935889**、MRR@10 **0.762501**；产品词法基线为0.668854/0.836556/0.635099，两组均300/300、零执行故障。模型选择来自独立32题验证，test没有用于调参；NIST -c -q逐题和总体复核通过。

这些是预算前候选排序和精确余弦参考dense的指标；实际OpenViking/HTTP小样本独立验证，不等同5,183条OV索引、最终证据或答案质量。本机选定模型排名段p50/p95约16.05/21.82秒，交互延迟仍需按目标硬件优化。完整原始排名、journal、置信区间、身份与失败记录见 [结果与证据](docs/development/generic-benchmark-20261106/RESULTS.md)。
