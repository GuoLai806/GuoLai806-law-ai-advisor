# AI 执行版：民法典 OCR、OpenViking 结构化索引与 Timely RAG

用户版见 [PLAN-FOR-YOU.md](PLAN-FOR-YOU.md)。开发状态截至 2026-11-11：已完成隔离实现、真实 PaddleOCR 64 页、1260 条官方目录树、真实 OpenViking 0.5.3 全量索引/回读/可见性验证、时间检索、API/原文回读、wheel 构建与安装检查。当前源码后端 503 项、前端 11 项通过。工作分支 `feat/versioned-legal-rag-20261105`，开发前 HEAD `87c64c4c2d5ca10fe95e71e1bec36193a3ba8312`。

本轮真实数据事实：

- PDF SHA-256 `4cef5e4a1e5f82c5a52e949093ba0cfec4f7290e78d04d837c292e2a62c4065e`，64 页。PaddleOCR 3.7.0 / PaddlePaddle 3.3.1，200 dpi，110,753 字符，0 空页，最低单行置信度约 0.875195。
- 1260/1260 条号边界恢复；1259/1260 与独立原生抽取逐条字面一致；1150 条双向完整包含。唯一复核条“第一千零八十五条”将“判决”识别为“决”，必须保留 `ocr_reconciled_review_required`。
- 国家法律法规数据库核实的标题、ID、发布机关、公布日、施行日和 1260 条目录树已保存。官方 PDF 下载端点返回 HTML 壳，未取得可信官方正文；正文来源单独标识为仓库 PDF hash，不能合并声称“官方直链封面/正文均已下载”。
- OpenViking `viking://resources/law-cn/civil_npc_tree_approved`：发送/内容回读 1260/1260，全量向量可见验证 1260/1260，匹配排名均 1，耗时 201.49 秒索引、30.28 秒验证；活动 revision 仅在验证后切换。
- 正式检索 30 题的标题已冻结：施行前 10、有效期内 10、历史法律问题 5、当前一般问题 5。BM25、OpenViking+BGE-M3、RRF、RRF+Timely 同一语料/candidate pool 对比。测试标签不参与在线排序或调 alpha。
- 正式结果：RRF 无时间 nDCG@10 0.487698 / Recall@20 0.5 / stale leakage 1.0；RRF+Timely 0.987698 / 1.0 / 0.0，平均检索 19.16 ms。这个数据集是按时间切片构造的小型定向集，不能作为公开通用榜或法律问答准确率。
- Qwen3-Reranker-4B 与 BGE cross-encoder 权重缺失，执行失败；未用历史 SciFact 成绩替代。生成模型未配置，chat 真实返回 evidence-only fallback。
- 后端 503 passed；前端 11 passed；wheel SHA-256 `589ee363e708029395d3a534cdc04ff58ced6aca26dcdda68faa845865ecec08`，安装后从 site-packages 导入 `evidence_rag`/`benchmarks`/`legal_rag` 且 CLI 帮助通过。Python/Node 原始日志、OCR 汇总、索引验证、正式结果、API 结果都在 `evidence/`。

保留的安全与法律边界：时间有效期规则不交给 LLM；过期/未来/未知版本不得生成；事件日优先。缺少有效期时返回 insufficient/unknown。结构路径、正文 hash 和 source locator 必须同源。写入 unknown 结果先 reconcile，不自动重放。公开库不提交 DB、模型权重、页图、Ocr 原日志或凭据。

仍未完成：独立人工法律问题集、多个真实历史 revision、后续修订/司法解释与过渡规则、可信官方正文下载、真实 Qwen/BGE cross-encoder 对照、生成问答质量、production 并发/容量。下一阶段需先补这些外部条件和数据，不应继续用同一 30 题调参。

## 目标、范围与可验证需求

目标是形成可开发、可评测、可维护的中国大陆民法典检索：PDF经PaddleOCR和高精度清理形成结构正确、可追溯的条文；OpenViking保存编/分编/章/节/条/款/项路径和正文；查询按事件发生日与版本有效期过滤、排序和引用。文档仅限中国大陆法律，默认事件日优先；时效必须由有来源的版本数据证明，不能用PDF创建时间或向量相似度代替。正式验收以时间有效性优先，随后才是nDCG/Recall/MRR与回答质量。

R1解析：输入仓库64页PDF或可替换扫描版，输出可重跑的法条级JSONL与页图/OCR中间产物；支持原生文字和PaddleOCR两条路径，生产选择必须由数据决定并保留失败。

R2清理：统一NFKC、中文条号、空格和全半角；合并句中断行，保留法律原文标点、否定、例外、条件、数值和单位；分离目录、页眉页脚、页码、编章节定位；不得跨条推断新正文。

R3验真：从正文第一次出现连续条号恢复，记录原始文字/OCR文本、清洗操作、条号、页码、坐标、行/字置信度、正文hash；目录未恢复视为失败；最终必须出现第一条至第一千二百六十条且无重号/空条/乱序。另与两份已存的政府转载逐条NFKC去空白对账，并记录非原始颁布载体验真边界。

R4结构：编、分编、章、节、条、款、项、附则形成父子/顺序关系；保留条号内`之一`，正文引用形成cites关系；完整条作为回答证据，款/项用于定位，不能把项碎片冒充完整法条。

R5索引：同一revision内Catalog稳定ID与OpenViking URI一一映射；URI包含corpus/legal/doc/version/category/chapter/section/article等稳定组件，metadata记录结构路径、content hash、页码、事件有效期、修订/取代关系、来源；完整原文和结构化字段同源，禁止只更新向量或只更新SQLite。任何import/reindex走journal、send/reconcile/validate、expected CAS activate；未知写入不盲重试。

R6时间模型：query保存`query_time`（服务器可审计的询问时刻）、`event_date`、可选显式时间窗；document/article保存`valid_from`、`valid_to`（开区间/闭区间语义显式）以及publication/insertion time。没有来源说明就保持unknown，不根据“最近导出”推断生效。时间过滤用规则，不交给大模型猜测。版本被后续条文的明确替代覆盖时，旧证据在覆盖日失效；适用保留、过渡、追溯、特别规定等不能只靠发布日期自动推断，必须带来源/备注或进入待核验。

R7时间评分：第一阶段沿用真实OpenViking+BGE-M3、BM25与现有RRF/模型重排；第二阶段只对候选按TimelyRAG思路做时间重排。有效性作为硬约束/强惩罚，时间距离作为软信号。借鉴TimelyRAG的查询事件时间、查询插入时间、文档事件时间、文档插入时间、期间距离、因果性、候选池归一化、时间粒度、缺失信号和动态权重；实现上使用可审计参数与显式的法律策略，不照搬默认值或覆盖语义。核心公式至少保留semantic、temporal、validity三部分，最终输出每项贡献。

R8回答：生成仅接收当前时间与版本都有效的完整条文；回答携带`valid_from/valid_to`、版本/修订依据、可能歧义、缺失事实。版本未核验时返回`applicability_unknown/needs_clarification`，不能把未过期的当前文本说成历史时间一定有效。事实不足与一般法律说明分开。

R9UI/API：现有API的`event_date`扩展为完整时间契约，新增时间依据与trace；默认循环本机身份边界不变。前端显示“事件日期/适用版本/有效期/时间筛除原因”，明确用户输入的时间是否被采用。不能把内部日期评分当法律结论，还需展示文档来源。

R10评测：固定开发集与正式集，全部包含有效版本gold与困难负例，覆盖同一法条多版本、有效期内/外、修订交界、追溯/例外、缺日期、来源不明、非现行历史事实、域外/其他法。真实民法典语料必须与开发数据分开；公开TimelyQABench只用于算法验证且注明合成/韩语/来源限制。历史120题只能作开发防回归，除非完成逐题复核和独立时间gold。

## 当前实现事实和需要修改的路径

已落地路径：`evidence_rag/catalog.py`支持条号、Structure、原文 hash、locator、temporal_versions 和官方 1260 树；`evidence_rag/ocr.py`负责 OCR 结果解析与逐条对账；`evidence_rag/legal_source.py`负责官方元数据、目录树、文件签名和来源失败记录；`evidence_rag/viking_index.py`保存结构元数据并通过 journal/send/reconcile/validate 闭环；`evidence_rag/retrieval.py`以有效期硬过滤加时间二阶段重排；`api.py` 与前端传递 `event_date/query_time/revision_selection/temporal`。OpenViking 的新时间衰减只可作辅助，法律效力由 Catalog 裁决。

## 技术选型和依据

PaddleOCR：官方3.7.0提供PP-OCRv6与PP-StructureV3，具备布局/表格/层级和坐标输出，适合扫描版与结构化。当前PDF原生文字层完整，开发应避免为了“复现技术栈”而忽略更强的原生抽取；PaddleOCR作用是回退、图像验证和未来复杂文档。固定具体模型、框架/Python/版本，保存官方来源与hash。不要把官方OmniDocBench分数改写为本项目实测。

OpenViking结构化索引：现有真实0.5.3全流程已证明scope/revision与原文回读；本方案继续用已验证后端，新增目录/元数据。目录语义提供范围检索与结构路径，但最终仍重排并回读完整法条。OpenViking时间衰减依赖写入时间，不等价法律生效期；不得仅靠它判定历史适用法。

TimelyRAG：论文与官方仓库`kaist-dmlab/TimelyRAG`提交`5213a0c1743f23fd52a258005c3887aaf7e8cd0f`；公开数据是合成、韩语法规，law集3000 queries/30000 docs，报告最大+28.6% nDCG@10和+19.1% Hit@10，不能直接迁移成中文民法典效果。官方实现使用QIT/QET/DIT/DET、时间距离、候选池归一化、gate、auto-alpha与grid best；`best-alpha`用gold labels才存在，不能在产品中使用。产品使用auto strategy，正式比较必须固定策略并避免用测试标签选参数。

## 外部依赖与数据取得

PaddleOCR版本、模型/框架适配需单独锁定。民法典后续修订版本需从全国人大法律法规数据库或其他可核验来源取得，保存HTTP最终URL、响应时间、发布机关、公布/施行日期、完整hash、页面/文本和替代关系；政府转载只能作辅助交叉，不能冒充原始载体。当前PDF来源未核验，不能改变PDF本身；新权威来源生成新revision。数据与OCR模型权重不入Git；只提交可公开的脱敏报告/定位和必要校验hash。公开边界仍排除企业/Pi资料/私人配置。

## 开发任务和依赖顺序

T1数据与来源契约（先行，阻塞正式activate）：定义版本、条、款、项、有效期与修订关系schema；为2020版填可核实公布/施行元数据，获取并核验后续版本或明确只做历史样例；产出source manifest、版本时间线、未核验清单。

T2OCR与清理（可与T3初期并行）：建双解析路径，逐页图像化、PaddleOCR输出、坐标/置信度、清洗规则、目录/页眉页脚隔离、断行合并；以原生文字与两个转载做自动逐条对账。验收1260条全量、每条原文和clean hash、失败明细；人工抽检结构与关键差异。

T3结构索引（依赖T1 schema，可与T2对接）：实现结构模型及OpenViking tree/URI/metadata/validate；确保Catalog、OpenViking和原文回读三方稳定映射。验收所有条都能按tree/scoped find找到，完整法条与款/项身份正确。

T4时间策略（依赖T1；可在没有后续版本时以构造fixtures先行）：实现时间解析、版本有效性、硬过滤/强惩罚、TimelyRAG二阶段重排、trace；产品不使用gold。验收边界/闭区间/未知/冲突/追溯、日期因果性和过期负例。

T5API/UI与回答（依赖T3/T4）：扩展请求时间和响应依据，显示有效期与被筛原因；生成门槛、grounding、revision/owner约束保持一致。验收真实SSE、会话、原始条文读回。

T6评测与复现（依赖T4/T5）：冻结开发与正式协议，保存每条候选、时间gold、错误、阶段延迟、模型/索引身份；与同候选池的普通重排A/B。正式报告分有效版本命中、过期错误、temporal nDCG/Recall/MRR、总体nDCG/Recall、回答引用和失败；小样本给区间并保留全部失败。

T7安装、构建、文档、评审与发布：独立环境安装、wheel/入口、真实服务、前端、回滚；双版文档、决策记录、原始证据、复现说明；普通提交推送并远端回读。不得改main/可见性/合并PR/外发资料。

## 验收门槛

功能：原生文本和OCR两条路径均可审计；PaddleOCR异常不导致静默错条；1260条完整；结构/索引/原文三方一致；时间过滤与重排可重复；旧版/新版/同日有效期/未知日期分别有证据。

时间正确性：正式时间题的有效版本Hit@K与Recall达到预定门槛，过期证据泄漏必须为0（对已知有明确有效期的题）；缺有效期证据不能被标valid。预定门槛在T6冻结前确定，禁止根据正式结果调目标。

检索：同一第一阶段候选池下，Timely重排与无语义/时间对照；报告总体nDCG/Recall/MRR和分切片，不得用历史SciFact 300/30或韩文公开数据替代本方案结果。算法提升需给出配对差及区间；若无提升或时间硬规则更重要，如实保留。

质量与性能：每条回答引用时点有效原文，不能仅凭时间排名声称法律正确；OCR/结构错误率、索引导入/验证耗时、检索/重排延迟、模型用量和缺报都保存。无法达到硬约束就缩小能力/返回待核验，而非隐藏失败。

## 已确认与工程默认

当前范围仅中国大陆民法典；用户确认自动抓取和自动检索，不增加发布审批流程。事件日优先；施行前没有已验证旧版本时返回 insufficient，不猜测。一般当前问题只有在版本有核验有效期时才作为当前文本使用；仍缺未来修订和过渡规定。

## 回退和禁止动作

新 OCR/结构化索引必须成为新 revision，旧 scifact 和 pending data amendments 保留；新索引未 validate 或时间来源不完整时不得切活动 scope。回退先停止应用写入，再切 Catalog/revision，不覆盖在写 DB。OpenViking unknown send 先 reconcile，不换 scope 绕过。公开提交不含 DB、模型、页图和凭据；业务数据仍留在本机。
