# RAG开发来由与实际实现

[当前用户版](PLAN-FOR-YOU.md) · [AI版](PLAN-FOR-AI.md)。本页从原调研和交付总记录摘出本项目内容；旧提案以现行实现为准，来源hash见evidence/history-sources.json。

## 2026-11-05旧问题

下表最后一列是当时建议，不保证全部法律专属功能现已实现。PDF结构/OCR、法律质量和模板数量仍需独立证据。

|编号|旧位置|发现|当时建议|
|---|---|---|---|
| L01 | `LegalAgent.py::_handle_consultation` | `retrieve` 和设置 context 的代码被注释；`consult(input, None)` | 新问答链路必须先形成 EvidenceBundle；不能在检索失败时悄悄退化成有法条依据的回答 |
| L02 | `server_text01.py::stream_chat` | 先等待完整 `process_input`，再逐字输出并 `sleep(0.03)` | 使用模型的真实异步 token 流；分别记录检索耗时、首个回答 token 和总耗时 |
| L03 | `server_text01.py` 的全局 `legal_agent` | 所有请求共享可变 AgentState 和咨询记忆，缺少用户及会话隔离 | 无状态 orchestrator 加按 user/session 定位的会话存储；请求内状态不放全局单例 |
| L04 | `PDFProcessor.py::process_pdf` | 每页独立分块；跨页法条可能拆开，清洗把换行合并 | 文档级结构解析，按编章条款项建模，再保留 PDF 页码与字符映射 |
| L05 | `HybridRAG.py::_linear_rerank` | BM25、向量、页码位置和关键词作线性组合 | 页码仅用于定位；替换为候选融合加 cross-encoder 重排 |
| L06 | `RLHFReRanker.py::_update_weights_from_feedback` | 正反馈增加两个手写权重，负反馈减少另一个；主 OptimizedRAG 路径实际调用 `_linear_rerank` | 命名为旧启发式排序；不称作训练后的 RLHF，不把未收集的反馈当成数据 |
| L07 | `VectorDatabase.py::query` | 以 `1 - distance` 表示相似度，未在创建集合处明确距离度量；随后候选内 min-max | 新接口显式声明 score_kind；融合使用排名，不跨不同分数域套同一阈值 |
| L08 | `DocumentGeneration.py::document_templates` | 当前注册 contract、complaint、legal_letter 三个模板 | 保留三种兼容入口；“10+ 模板”需补其他来源或改写简历 |
| L09 | 启动及健康检查 | `initialize()` 的布尔结果未成为成功日志和 readiness 的强制条件；还存在两个 RAG 实例 | 只创建一个 RetrievalService；ready 需要活动版本索引和模型能力实际就绪 |
| L10 | 依赖与仓库资料 | 旧依赖固定与未固定混用；默认 PDF、训练资产和历史评测报告未构成可复现交付 | 新环境独立锁依赖，先补语料清单和评测；不在这轮安装或改线上环境 |

旧源码固定在 [689f2a5](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/tree/689f2a5bf4b853dfe93ac26802970f8996f993c8)。例如咨询LegalAgent.py、流式server_text01.py和排序HybridRAG.py可从该树读取；这是源代码检查结论，不把所有路径写成用户事故。

## 后续开发与决定变化

|阶段|实际工作|为何采用或修订|
|---|---|---|
|11-05→06|版本化通用核心、身份隔离、真实SSE、来源回读|用户允许通用方向；法律兼容调用同一核心，不强加条号规则|
|11-06参考排名|公开SciFact全库，候选模型对比，BM25+BGE-M3+RRF+Qwen|保存排名和失败；参考dense不能代表真实OV用户路径|
|11-06真实OV|实际resources/find与云模型SSE、原文和会话回读300题|用户明确指定OV实际体验，31失败全部保存|
|11-07回答质量|可选grounded结构和字面摘录、有限空输出重试；正式30|解决证据不足和引用；用户将规模缩为十分之一|
|11-07 bvio对照后|独立连接复用与资料预览，直接/预览各30|记忆查询与证据问答用途不同，没有企业整体胜负；局部复用有数据|
|11-08页面0.2.3|无令牌本机身份、资料/最终答/延迟、停止状态|用户要求更清楚；451+10和安装网页通过，正式题未重跑|
|11-09→10|进度/交接文档，服务按原配置恢复后健康200|文档和运行快照不计作质量升级|

## 规格到实现

核心类名与接口以evidence_rag代码为准；旧三路法律召回和法律时间过滤不是general默认。原门槛may_generate现为generation_allowed。现行导入是完整revision，已有同步stage/jobs；异步worker完整生命周期另做。原文/定位存储不保证复杂长PDF完整向量覆盖，SciFact测试单位是标题摘要。参考排序、真实OV、策略30与最后UI各自绑定版本，详见[DECISIONS](DECISIONS.md)与[阅读顺序](READ-FIRST.md)。
