# 需求规格：民法典 OCR、OpenViking 结构化索引与 Timely RAG

[用户版](PLAN-FOR-YOU.md)是本页的同一组决定；本页是开发依据。状态为需求方案阶段，2026-11-11 核实；本轮不实现业务代码，不安装 OCR 依赖，不改活动资料库。目标仓库：`GuoLai806/GuoLai806-law-ai-advisor`，工作分支 `feat/versioned-legal-rag-20261105`，核对 HEAD `0d14a09773c54c9432475a4e357bbe893304d97b`。来源 PDF：`main-system/中华人民共和国民法典 - 中华人民共和国最高人民法院.pdf`，64页，SHA-256 `4cef5e4a1e5f82c5a52e949093ba0cfec4f7290e78d04d837c292e2a62c4065e`。

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

`evidence_rag/catalog.py`已有法律条解析，返回1260条、案例约2126个候选款项（历史计数）；但本次核验发现正文存在跨行断字，例第一条“发展要/求”，第1260条“合同/法”。现有条号是分离unit，目录解析依赖正文最后一次连续`第一条`；已从当前库确认1260条，但仍需重新运行新解析器做逐条结构检查。当前manifest是`repository-unverified`、source_url为空、effective_from/to为空，不能直接作为权威现行法律激活。

`evidence_rag/cli.py`法律导入直接用PyMuPDF `get_text()`；`Catalog.stage`全局解析条号并建relations。需要增加显式parser/cleaning version、页/坐标/置信度、OCR原文与清洗后文本、结构表。`evidence_rag/retrieval.py`使用BM25+OpenViking+RRF+Qwen；`event_date`只影响query_plan与法律profile提示，未参与候选过滤/排序。`effective_from/to`取的是整个revision manifest，不是逐版本/逐条有效性。`evidence_rag/viking_index.py`目前给整个unit目录和正文；需要设计tree与L0/L1/L2层级，确保文档/条/款可追溯到完整正文。OpenViking当前已有`find/search`的`events_time_decay_protection`和`since/until`接口（`work/research/OpenViking/docs/en/api/06-retrieval.md`），但项目目前仍调用`find`；它们可以考虑作为辅助，法律有效期仍由Catalog规则裁决。

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

## 待确认与工程默认

按目前环境建议：版本范围直接做真实中国大陆民法典时间线；默认事件日优先，一般规则可用当前版；正式评测时间有效性优先；公共服务需有来源/内容核验。仍未确认、需用户决定的只有两点：是否公开服务范围内只处理中国大陆民法典；现行/历史版本是否必须经法律专业人员复核后才可标有效。若未回答，开发可先用fixtures、隔离revision与`applicability_unknown`完成技术链路，不把未核验内容发布成权威法律结论。

## 回退和禁止动作

新OCR/结构化索引必须成为新revision，旧scifact/当前民法典pending revision保留；新索引未validate或时间来源不完整时不得切活动scope。回退确认应用进程停止后切回原有Catalog与revision，不覆盖在写数据库；OpenViking unknown send先reconcile，不换scope绕过。本轮不安装依赖、不运行OCR、不调用模型、不停服务、不改业务数据库，等待你对本方案确认后进入实现。
