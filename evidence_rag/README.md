# 通用证据检索核心

`evidence_rag` 提供带版本和来源的文档检索；`legal_rag` 是同一核心的法律兼容入口。通用路径默认不应用法律条号或事件时间规则。数据格式与本机运行方法见[开发记录](../docs/development/general-evidence-rag-20261106/WORK.md)。

## 网页和真实流式回答

启动 `evidence_rag.api:app` 后，`GET /` 提供资料问答网页。输入操作者配置的访问令牌，提交问题，页面接收真实 `/api/v2/chat/stream` SSE；完整生成并通过引用编号检查后才记为完成。页面能展开原文，停止时撤回未完成草稿。引用编号有效不代表每句话受到原文支持。

`EVIDENCE_RAG_DEFAULT_EVIDENCE_ITEMS` 控制正常请求和网页的默认证据数，范围1..60；`/api/v2/models.runtime_identity` 返回实际默认、提示词hash、模型与预算身份。`EVIDENCE_RAG_STORE_CONVERSATIONS=1` 显式启用完成答案保存，按访问身份绑定会话；通过 `/api/v2/sessions/{id}/turns` 回读。

操作者可设置 `EVIDENCE_RAG_MODEL_THINKING=enabled|disabled`、`EVIDENCE_RAG_MODEL_REASONING_EFFORT=low|high|max`、`EVIDENCE_RAG_MODEL_TIMEOUT` 和 `EVIDENCE_RAG_MAX_OUTPUT_TOKENS`。模型是否支持这些参数取决于provider；本轮验证使用DeepSeek Pro，其thinking模式忽略temperature。

空响应重试默认关闭。仅在首次thinking为enabled、输出有明确上限时，可配置 `EVIDENCE_RAG_MODEL_EMPTY_RETRY_THINKING=disabled`、`EVIDENCE_RAG_MODEL_RETRY_TIMEOUT` 和 `EVIDENCE_RAG_MODEL_RETRY_MAX_OUTPUT_TOKENS`。首次完全没有文本且超时、无完成标记或空响应时，同证据再试一次；部分文本、连接错误、限制终止和取消不触发该策略。SSE保留每次调用、重试原因及已报告usage，未知用量不补零。生成失败的脱敏错误事件只含类型、阶段和可用HTTP状态，不含上游正文或凭据。

真实OpenViking用户路径评测、当前状态和验收口径见 [用户版](../docs/development/openviking-user-flow-20261106/PLAN-FOR-YOU.md) 与 [执行版](../docs/development/openviking-user-flow-20261106/PLAN-FOR-AI.md)。

## 证据预算和生成

默认 `budget_mode="tokenizer"`：需要配置目标生成模型对应的本地 tokenizer 和 chat template；未配置时仅返回检索证据，不生成回答。是否真正可用仍取决于 tokenizer 与服务端模板是否一致。

仅用于受控互操作试验，可在请求中显式设置 `budget_mode="conservative_bytes"`，同时由部署者设置：

- `EVIDENCE_RAG_ALLOW_CONSERVATIVE_BYTES=1`
- `EVIDENCE_RAG_MODEL_CONTEXT_WINDOW`：模型的实际上下文窗口。
- `EVIDENCE_RAG_FRAMING_RESERVE`：为未见的消息模板和协议开销预留的数量。
- `EVIDENCE_RAG_OUTPUT_RESERVE`：为回答预留的数量；生成器必须显式限制 `max_output_tokens`，且不超过该值；从环境配置的生成器使用 `EVIDENCE_RAG_MAX_OUTPUT_TOKENS`。

该模式按所有实际消息内容的 UTF-8 字节数选择**完整证据单元**，再保留上述 framing 与输出余量。返回 `state=conservative_bytes`、`content_bytes`、`omitted`、`exact_tokens=false` 和非形式保证说明，`used_tokens` 为 null。字节数与保留量是保守启发式，不保证覆盖未知 tokenizer、服务端 chat template、隐藏提示或协议开销，不能称作精确 token 统计或生产校准。某一完整单元超出预算时整单元省略，不在段落中间截断；被省略的 ID 在响应中列出。法律 profile 不接受该试验模式。

服务默认还要求实际可用的检索组件。一般问题的生成需要独立标定的重排阈值；未设置时只返回证据。精确指定 document_id 的原文查询按已有 source lookup 规则处理。生成输出的 `[E1]` 等引用只做结构检查；`claim_support=not_verified` 明确表示没有完成语义支持验证。服务端如返回 token 用量，v2 SSE 会发 `usage` 事件，普通 chat 响应会返回 `usage`；普通/chat按attempt保存最后累计usage并累加已报告调用，同时返回usage_attempts、generation_attempts、generation_retries、observed_answer_models和usage_complete。全部调用有报告为provider，只有部分为provider_known_attempts，已尝试但无报告为unavailable，未调用生成为not_generated；异常和引用拒绝仍保留已知用量。无attempt的旧用量事件归入1，仅文本provider保持兼容且用量未知。不用字节数冒充模型真实用量。需要 provider 支持时可使用 `EVIDENCE_RAG_STREAM_INCLUDE_USAGE=1` 请求流式 usage。

`/health/ready` 的严格就绪判断仍要求 tokenizer 等依赖；即便显式启用保守字节模式，也不会因为该近似模式单独标记完整 ready。此开关仅说明可尝试近似预算，不代表精度、法律适用或安全验收。
