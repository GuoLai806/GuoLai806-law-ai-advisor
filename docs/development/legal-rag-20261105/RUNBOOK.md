# 隔离运行与模型实验契约

工作目录为法律仓库根目录。先在隔离 Python 3.12 虚拟环境安装 `requirements-v2.txt`；这些是未锁定直接依赖，测试环境版本见 `evidence/test-environment.txt`，不是可复现生产锁。默认不加载旧项目的 torch 或 LangChain。

## 服务环境变量

- `LEGAL_RAG_DB`：独立 SQLite 路径；不指向旧业务数据库。
- `LEGAL_RAG_AUTH`：JSON 对象，键为随机 bearer token、值为服务端用户标识；仅通过本机安全环境注入，禁止写入源码或证据。无配置或错误凭据均拒绝。用户身份不接受请求体自报。
- `LEGAL_RAG_ADMINS`：逗号分隔已配置用户标识，只有这些用户可提交和查询导入。
- `LEGAL_RAG_INGEST_DIR`：管理员可导入 PDF 的目录；接口只接收此目录下文件名，限制 50 MB，所有导入强制 `pending`，无 API 能声明审核通过。
- `LEGAL_RAG_STORE_CONVERSATIONS=1`：可选。默认不存对话正文；启用后普通 chat 和完成的模型流把请求、回答和固定 revision 保存到 SQLite。SSE 仅证据展示或中断的未完成流不保存正文。数据库没有应用层加密，应由部署方配置文件权限、加密、保留期限和备份；不要把用户数据库提交。session 所有者始终持久绑定，历史查询再次检查身份。当前历史不用于自动查询改写。
- `LEGAL_RAG_LEGACY_ADAPTER=1`：启用旧 `/api/chat`、`/chat`、`/api/chat/stream` 适配；默认关闭，仍必须鉴权。未给 session_id 时从服务端用户标识计算稳定 session ID；旧静态页面不会自动附认证头，需要改造或可信认证网关。模型类型字段只为兼容响应，不能选择模型，也不走旧 Agent。
- 可选 dense：`LEGAL_RAG_OV_URL` 和 `LEGAL_RAG_OV_SCOPE`（scope 末段必须等于请求 revision）；可通过 `OPENVIKING_API_KEY` 注入 X-API-Key；实际 OV 服务尚未联通。按 `viking_index` 的 send、reconcile、validate 操作生成映射，未知结果不可自动重试。回读匹配不等于向量可检索。
- 可选重排：`LEGAL_RAG_RERANKER=bge` 或 `qwen`，可配 `LEGAL_RAG_RERANK_DEVICE` 和 `LEGAL_RAG_RERANK_DTYPE`（float16/float32），仅加载本地固定 revision；模型依赖或权重缺失时启动失败。
- 可选预算：`LEGAL_RAG_TOKENIZER` 和 `LEGAL_RAG_TOKENIZER_REVISION`，必须是与实际生成服务 chat template 一致的本地 tokenizer；否则生成降级。
- 可选生成：`LEGAL_RAG_MODEL_URL`（OpenAI 兼容 `/v1` 根地址）、`LEGAL_RAG_MODEL_NAME`、`LEGAL_RAG_MODEL_KEY`。仅当来源查询或一般说明不需要具体案件适用判断、证据完整、tokenizer 已测量、dense 验证通过且 reranker 可用时请求生成；输出实时 delta 在最终检查前标为 pending，结构引用校验不证明主张有法律依据。个案无日期时追问，有日期仍标为适用性待审核并只展示证据。密钥不写日志或仓库。

启动：`uvicorn legal_rag.api:app --host 127.0.0.1 --port 8000`。仅监听 loopback；readiness 检查活动 revision、FTS 数量、向量校验状态、OV 健康、模型列表与加载状态；不能代替真实推理和法律质量验收。静态 bearer 不具备令牌轮换、过期、集中授权、TLS 或速率限制，生产使用需真实身份网关与安全评审；禁止公开监听。

## 本地模型实验

先准备固定 revision 权重，安装 sentence-transformers、transformers、torch，并冻结硬件、软件、模型和 corpus；该脚本使用 local_files_only，不隐式下载或调用收费接口：

```sh
python -m legal_rag.model_experiment --db /tmp/law.sqlite3 --revision repository-unverified --dataset docs/development/legal-rag-20261105/synthetic-300.json --embedding bge --reranker bge --device mps --dtype float16 --batch-size 2 --output /tmp/bge-bge.json
```

将 `--embedding` 和 `--reranker` 各取 `bge`、`qwen`，得到四组。默认按数据前 300 条运行；生成全部 corpus 向量后逐题融合 exact、BM25 和 dense 的名次，再对最多 60 条完整款项重排，输出每题模型原始分数、排名、召回、nDCG 和检索耗时。先小规模验证 GPU/CPU 内存、运行时间、模型 API 和权重身份；Qwen 4B 模型有较高资源消耗。数据集为源文摘录，评价数字不能用于真实质量声称；正式测试必须使用人工独立标注数据。无任何模型实测结果时不得把脚本存在当作实验成功。

真实来源的启用还需要操作员离线核验：先在导入 manifest 中登记官方 `source_url`，核对原始文件 hash、完整性、版本与页码，然后运行 `python -m legal_rag.cli approve-source --db PATH --revision REV --reviewer NAME --evidence '具体核验依据及定位'` 留下记录，再完成全部 OV 回读和向量可见性验证，最后使用 `activate --expected` 做 CAS。这里保存的是操作员的核验声明，系统不会自动证明官方来源；无法核验时不要调用 approve。供单测用的 `source_kind=synthetic` 可在隔离库跳过真实 OV 检查，绝不能用作真实语料的绕过方式。

## 官方来源核验流程（尚未完成）

候选官方 PDF：[全国人大法律法规数据库民法典](https://wb.flk.npc.gov.cn/flfg/PDF/bd53dd912c1048f2aecbaa229238334b.pdf)。在隔离目录下载后保存下载时间、最终响应 URL、文件 SHA-256 与完整 HTTP 记录（去除认证信息），核对发布主体、公布/施行日期、是否含全部 1260 条及附则、文本与页码、有效性和再分发范围。对照仓库 PDF 的 hash 仅能发现文件是否相同，不能单凭 hash 一致认定权威性。把核验过的 URL 写入新 manifest 并记录人工证据；任一项未核实就保持 pending。未完成这些步骤前，不能声明已经核验官方原文。

2026-11-05 UTC 首次下载尝试发生 TLS 证书过期校验失败，未绕过验证、未取得官方 PDF；详情见 `evidence/official-fetch.json`。来源、完整性和再分发范围仍未核验。

一般概念的模型解释额外要求 `LEGAL_RAG_RERANK_THRESHOLD`，其值必须来自与当前模型 revision、语料及验证集匹配的离线标定；缺省时只展示证据。此项目尚无真实标定阈值，不得把随意填入的值称作已验证。明确法条原文查询仅返回指定条号；模型输出仍需核验，不把原文查询当成案件适用结论。

`requirements-v2.lock` 是此次 Python 3.12 测试环境的运行依赖版本快照，不含 wheel 哈希、平台解析或模型依赖，也未在干净环境重新安装验证；不能冒充完整可复现生产锁。生产环境需根据目标系统重新解析、带哈希锁定并安装验证。

随后对两个政府转载页面完成严格 TLS 下载和逐条机器交叉核对，见 WORK.md 与 `evidence/official-crosscheck.json`。两页和本仓库 PDF v4 提取的 1260 条经 NFKC 和去空白后逐条一致；此结果不证明它们是全国人大原始颁布载体，也不代替人工确认修订、时效、文本格式与转载许可。不要据此直接 approve。

模型实验先完成所有 corpus/query embedding 并释放 embedding 模型，再加载 reranker；`--batch-size` 同时限制两侧。每题 latency 是先测 query embedding 再加稍后排名耗时，不包含模型加载或语料向量化，不是连续端到端延迟。失败题分别记录并标记，均值不含未成功计分项，不能掩盖失败数。Qwen 的实际模型及模板路径仍未经过权重实测。
