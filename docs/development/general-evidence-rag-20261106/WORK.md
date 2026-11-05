# 通用证据检索：实现与验证

本次按用户新范围将核心实现放在 `evidence_rag`，默认 general；仓库名称保留。`legal_rag` 是薄兼容入口，复用同一套 SQLite、FTS、OpenViking、融合、重排和证据代码，默认 legal。历史法律文件、旧数据和证据均保留。当前候选仍是开发版本，不表示生产可用或已通过公开数据集验收。

通用输入支持 UTF-8 JSONL、文本/Markdown 和 PDF。JSONL 每行至少包含字符串 `_id` 与 `text`，可带 `title`、`source_url` 和 `metadata`；这些 ID 映射到稳定 `unit_id`，检索响应提供 `document_id`，可与公开数据集 qrels 对齐。全文作为一个可回读单元，段落以单独带 hash 和字符范围的 parts 保存并建立 FTS5 索引。普通问题返回匹配的完整段落，附带全文 hash、段落编号和是否还有更多内容；按 document_id 精确查看时回读全文。PDF 附页号范围。对超预算单元不截断，只在证据预算无法容纳时省略并标识。对任意输入进行严格文本与权限核验仍是调用者的责任；`provided` 仅表明输入来自提交者，**不是官方或人工核验**。

```sh
python -m evidence_rag.cli ingest --db /tmp/evidence.sqlite3 --input /path/to/corpus.jsonl --corpus-id demo --revision r1
python -m evidence_rag.cli activate --db /tmp/evidence.sqlite3 --revision r1
EVIDENCE_RAG_DB=/tmp/evidence.sqlite3 uvicorn evidence_rag.api:app --host 127.0.0.1 --port 8000
```

第一次 activate 的 `--expected` 可省略；已有活动版本时应传现有值，CAS 防并发覆盖。法律版使用 `python -m legal_rag.cli` 或 `uvicorn legal_rag.api:app`；法律资料仍需人工审核、来源记录和密集索引检查。无活动语料时 HTTP 检索不可用。

认证由服务端 `EVIDENCE_RAG_AUTH` 中 token 到用户身份的映射提供，管理员名单为 `EVIDENCE_RAG_ADMINS`，管理员目录为 `EVIDENCE_RAG_INGEST_DIR`；无身份配置默认拒绝。服务只用于本地隔离测试，静态 token 不是生产认证方案。`LEGAL_RAG_*` 仍可作为旧配置回退。通用检索不触发法律名/条号、事件日期或跨期规则；只有显式 legal profile 才启用。默认不保存会话正文；开启可选持久化前需自行保护文件与保留策略。

普通用户请求示例：`POST /api/v2/retrieve` 带 bearer、`query` 和 `session_id`，可选 `document_id` 精确读取、`corpus_revision`、`max_evidence_items`（1–60）和 `max_evidence_tokens`。不同来源的原始分数标明各自含义，RRF 按名次融合，不跨分数域当概率。未配置真实模型/索引时会降级，readiness 不会假报完整生成能力；通用回答也必须有完整证据、真实 tokenizer 和经单独校准的重排阈值。现有样本没有人工批准的阈值，默认不生成一般结论。

## 证据和限制

- `evidence/tests.xml`：原法律兼容及新增通用测试共 62 项通过，包括 JSONL、文本、两页 PDF、profile 区分、API 鉴权、CAS 和 scope 拒绝；本机 hostname 已脱敏。
- `evidence/trace-revision_d8505d20bb7e.json` 和 `evidence/reconcile-smoke.json`：真实 OV 0.5.3 + BGE-M3 在本机对两条英文合成文档进行 `vectors_only` 导入、L2 hash 回读、rag scope 检索和 CAS 激活。首次复用输入时幂等返回旧 revision，后续新版本已写入但 CAS 因旧版本活动而拒绝；检查状态后以正确 expected 激活，**没有重放已写资源**。原始失败与恢复都保留。
- `evidence/http-smoke.json`：真实本地 uvicorn 调用通用 profile，含实际 OV dense 召回、未认证 401 和缺少模型组件的 readiness 503。此测试是合成语料，不是公开基准质量或业务验收。
- 当前只有两个合成文档的通用实机集成；真实公开语料、qrels、端到端质量、性能和 Qwen 路径交由独立 benchmark 工作验证。没有把人工未核验题或假向量冒充评测。通用源的数据删除、敏感资料隔离、多租户生产身份和分布式并发不在当前实现保障内；不同用户默认共用活动公共语料，私有材料不能直接放入该共享索引。
- HTTP 导入当前同步解析后返回 202 的 indexed job；没有完整异步失败/取消工作流。非法律通用源允许 `provided` 激活供本机词法检索，但不表示向量索引或完整服务 ready。真实 OV 导入仍需逐单元日志、回读和可见性校验；已有未知投递不能盲目重放。

段落索引和 scope 守卫加入后，以新合成 revision 重新执行真实 OV 和通用 HTTP 冒烟，结果见 `evidence/trace-revision_df00d6f6d6ae.json` 与 `evidence/http-smoke.json`；依然不是公开数据集或用户验收。

共享 `rank_candidates` 同时供产品和公开基准调用：词法/密集各 top40、RRF k=60、候选最多60、同一段落重排；基准排名在证据预算前，应用展示在预算后，二者不能混称。通用实际 OV L2 与模型离线 embedding 输入均为 title + 换行 + text，不包含文档 ID 或 qrels；索引 manifest 固定该序列化身份，旧 body-only 通用索引需重建新 revision。法律资料仍保持原文 L2。

空标题文档此前会把内部 ID 误加入词法索引，已修复并增加回归；embedding、rerank 和 FTS 都不再把 doc ID 当作文本。通用 parser 身份升级为 `generic-documents-v2`，v1 通用索引需重新导入新 revision；法律 v4 保持兼容。真实 OV 与 HTTP 在 v2 的新合成 revision 重跑通过，见 `evidence/trace-revision_4f00bca1e710.json` 和 `evidence/http-smoke.json`。

可选通用试验模式 `conservative_bytes` 需要请求与服务双方显式启用，并指定模型上下文、framing 与输出保留；按完整段落选择，输出字节与非精确保证说明；默认仍要求 tokenizer，法律 profile 不使用该模式。流式接口可返回 provider 原始 usage；未收到则标 unavailable。68 项测试通过，其中包含超长单元整段拒绝、profile 边界和与 choices 同包 usage 的 mock 协议；本机 BGE tokenizer 合成样例对比见 `evidence/byte-budget-tokenizer-fixtures.json`。没有由本任务发起任何付费云模型调用。

公开启动路径已统一：根 `README.md` 和 `pyproject.toml` 提供通用及法律兼容 CLI；旧 `main-system/server_text01.py` 直接返回共享 legal app，不再导入旧全局 Agent；容器使用仓库根构建上下文和通用 API，本机端口与独立卷，未内置权重或凭据。`evidence/tests-public-entry.xml` 记录 72 项通过，覆盖旧入口不加载旧 Agent/torch、CLI 和 Compose 结构。`evidence/public-entry-build.json` 记录 wheel 构建、wheel zipimport 和 `docker compose config --quiet`；没有安装包、构建镜像或启动 Docker。`constraints-runtime.txt` 只是此次环境版本快照，待主 agent 在最终环境锁定和复验。

内存注入 probe 已用 loopback 假 provider、假重排与假 dense 通过真实 v2 HTTP 管道，返回模拟 usage，见 `evidence/cloud-probe-local-fake.json`；这不是云模型调用或质量验证。外部云调用与凭据由主 agent 单独操作，本子任务没有读取凭据或发出付费请求。

受控字节模式现在对显式全文查询按原有段落拆成完整证据，分别保留段落 hash/偏移及全文 hash；超预算段落整段列入 omitted。若任何所需段落省略，当前生成保护会继续保持证据展示状态。
