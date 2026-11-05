# 技术决定与实测范围

| 选择 | 备选与原因 | 证据或限制 |
|---|---|---|
| SQLite WAL + FTS5 | 首期单机、语料规模较小；避免引入额外检索集群；版本与原文可事务保存 | 本机 PDF 1260 个解析条目、合成查询；并发与崩溃恢复未实测 |
| jieba + FTS5 BM25 | 原生 unicode61 不足以作为中文分词；比直接字符串匹配更适合词法候选 | 实际对合成条号查询运行；词典版本未锁定，无法说明业务精度 |
| PyMuPDF 文档级文本 | 页面独立切块会拆断条文；对每页保留字符区间 | 跨页合成 fixture 通过；真实 PDF 字符提取与页码尚未逐条核验 |
| 精确条号 + RRF | 融合排序名次，避免直接把异源分数当同一概率；明确引用优先保留 | 三种实际执行的消融数据在 evidence；只有 3 个合成条号样例 |
| OpenViking HTTP v0.5.3 | 通过独立服务集成，而非复制 AGPL/私有源代码；资源 scope 和服务器结果均检查 | v0.5.3 `search.py` 的 FindRequest 契约；已在本机真实 OV 服务验证一条合成资源的写入、回读及 dense find；法律质量未实测 |
| BGE/Qwen CrossEncoder | 调用官方 sentence-transformers CrossEncoder 而非假分数；固定 revision，运行时必须本地有权重 | BGE 已完成合成真实推理；Qwen 尚未实测；模型通用指标不能代替法律评估；依赖未锁 |
| FastAPI + JSON SSE | 独立旧服务依赖，事件可区分证据与后续核验；模型包装读取真实上游 delta | API 已可连接可配置模型；mock 验证协议不等于真实流或断连验收 |
| 服务端 token → 用户映射 | 不信任客户端声明的 user_id；session 首次绑定用户 | 简易开发用途，缺少完整身份生命周期和生产安全验收 |
| 全文条款优先 | 不在例外和否定句中截断 | 已提供真实 tokenizer 接口，但尚未验证目标模型与服务端模板的一致性 |

OpenViking 接口参考：<https://github.com/volcengine/OpenViking/blob/v0.5.3/openviking/server/routers/search.py>。Qwen CrossEncoder 用法：<https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/22e683669bc0f0bd69640a1354a6d0aebcfeede5>。目标模型固定版本见 `legal_rag/retrieval.py`。所有选择均待实际来源、硬件和质量验证，未达成高精度或生产可用结论。

## 后续增量与验证边界

模型 tokenizer 使用 chat template 计算消息 tokens，不把字符数称为 token；缺少配置时只能展示证据且不能生成。上游 SSE 模型 chunk 已连接服务层并保留核验状态，协议只用模拟端点测试。OV v0.5.3 资源接口使用临时上传、指定目标与等待完成，日志在派发前记 unknown，单独用原文 hash 回读核对，再用受限 find 检查每条向量可见性；上述已在隔离真实 OV 服务用一条合成资料验证。官方协议源码：[资源](https://github.com/volcengine/OpenViking/blob/v0.5.3/openviking/server/resource_ingest.py)、[读取](https://github.com/volcengine/OpenViking/blob/v0.5.3/openviking/server/routers/content.py)。300 题源文摘录集存在严重数据泄漏，仅为工程冒烟，不能推断真实质量。

| 补充选择 | 原因与取舍 | 验证边界 |
|---|---|---|
| 搜索模式 jieba | 既索引长词也索引组成词；“合同纠纷”检索可以召回含“合同”的条文 | 修复 fixture 后通过；300 条源文派生集的关键词切片 nDCG@10 约 0.681，表明候选排名仍需改进 |
| 条号规范化 | 让中文、全角/半角数字与“之一”统一身份，同时保留原始标签 | 单元测试覆盖；不含复杂法律名称消歧或跨文档版本选择 |
| 款项窗口重排 | 每个完整款项独立打分，超出 4096 token 拒绝而不是静默截断 | 仅替身 tokenizer 验证；Qwen chat 模板额外 token 需要真实模型核对 |
| 三类查询 | 条文查看无需案件日期，一般概念与具体个案分开；个案日期/过渡规则不全不能得出适用结论 | 规则 fixture 通过；语言识别能力有限、无法替代律师判断 |
| 一般解释阈值 | 不把任意重排分数当概率；默认关闭一般问答，阈值需由验证集标定 | 未有人工集和任何真实阈值，不能宣称通过 |
| 未审核候选关系 | 提取明确条号引用并保留偏移，供后续审核；不根据正则自动扩展法律依据 | 真实仓库 PDF 提出 53 条候选；包含跨法律误关联风险，未人工核验 |
| 可选持久化 | 默认不保存敏感会话正文，显式启用后按认证主体绑定 session | 测试进程重启后仍可读，跨主体拒绝；不含应用层加密/生产会话治理 |
| 版本快照 | 用 `requirements-v2.lock` 记录本次直接及主要间接依赖版本 | 没有哈希、跨平台解析、干净环境安装或模型依赖锁，不是生产锁 |

真实 OV 以 `vectors_only` 索引原文，不需要伪造 VLM 摘要；一个合成资源的实际 HTTP 轨迹见 `evidence/ov-real-smoke.json`。首次请求因系统代理拦截本地端点超时，未重放原任务；第二隔离实例明确配置本地免代理后成功。BGE MPS f16 双模型合成运行见 `evidence/bge-real-smoke.json`，Qwen 模板长度和模型实测待完成。
