# 通用检索技术选择和边界

| 决定 | 理由和备选 | 实际证据与限制 |
|---|---|---|
| 单一 `evidence_rag` 核心 + `legal_rag` 兼容层 | 共享检索与证据组装，避免两套排序器随时间分叉；法律政策通过显式 profile 启用 | 原 45 项法律测试及新增通用测试通过；旧调用、旧数据库、旧 CLI 均保留 |
| SQLite WAL + FTS5 + jieba 搜索分词 | 单机无需集群；中英混合入库和查询一致。英文经 FTS5 unicode61 索引 | 合成英文与中文回归通过；不是所有语种的最优分词或生产性能证明 |
| 文档全文 unit + 段落 parts | 保持公开数据集文档 ID 与 qrels 映射；段落另建 FTS 索引以返回完整相关段落，重排同样使用段落 | PDF 两页与文本定位测试；超长单个段落或全文预算不足会拒绝/省略，尚无复杂版面 OCR |
| OpenViking 0.5.3 HTTP + `vectors_only` | 独立后端，dense 使用真实 BGE；不依赖伪造 VLM 摘要，scope 含 corpus 与 revision | 两个合成文档 real engine 写入/回读/查找通过，保存请求响应；不是完整公开数据集对照 |
| RRF + 独立 CrossEncoder | 名次融合避免混用异源数值；BGE/Qwen 可换且权重 revision 固定 | BGE/MPS f16 三条法律合成 smoke；BGE/Qwen四组候选、选定配置v3/v4真实32题均已完成，正式300题独立运行 |
| 认证从服务端 token 映射 | 客户端不能自报身份，会话 ID 首次绑定用户 | 测试跨主体拒绝；共享公共语料不构成私有索引、静态 token 不构成生产身份系统 |
| 通用 source `provided` | 用户提供文本可在本机建立可检索 revision，而不虚称官方验证；法律 profile 保持额外核验门槛 | 仅使用明确标记的合成文件；外部来源的许可、真实性和时效需另行审核 |
| API readiness 与证据预算 | 组件缺失或索引不一致不报完整 ready；tokenizer 不可用时仅展示检索证据 | 真实 HTTP 合成 smoke 检查 503，真实生成模型synthetic SSE互操作已完成，但未验证目标云tokenizer/template精确对齐 |

变更是通用基础设施方向；32题验证不代表正式300题已经完成，最终数据、各算法消融、原始逐题结果和许可来源以独立 benchmark 记录为准。

公开基准和产品使用同一 `rank_candidates`、`document_embedding_text` 和 `reranker_document_text`，避免独立算法冒充产品结果；公开基准的 exact cosine 是向量后端对照，真实 OV 性能与质量仍要单独测量。

## 可选字节预算（互操作实验）

精确模型 tokenizer/chat template 仍是默认。为无法取得目标 tokenizer 的短样例，可显式启用 `conservative_bytes`：完整消息内容按 UTF-8 字节计数，加部署者指定的 framing/output reserve 并受 context_window 限制。它不依赖伪 token 估算，也不是所有 tokenizer 或未知 provider 模板的形式化上界；不得标成精确 token 或生产校准。法律 profile 拒绝该模式。生成请求同时显式限制最大输出，不超过输出预留。

本机 BGE-M3 固定 revision tokenizer 的 5 个合成字符串和一组完整消息，内容 token 数均不大于其 UTF-8 字节数；原始统计见 `evidence/byte-budget-tokenizer-fixtures.json`。这只验证该 tokenizer 和这些字符串，不含特殊 token/服务端模板，也不能外推到目标云模型。流中实际 provider usage 仅在服务端返回时记录；未返回就标记 unavailable。没有真实 provider 请求时，不宣称使用量或语义正确。
