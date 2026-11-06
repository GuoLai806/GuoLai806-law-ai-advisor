# OpenViking 用户流程评测：执行依据

状态：正式300题已完成（2026-11-06T08:09:36Z–15:07:29Z），冻结源码7231a1fc00a20f5b311e083e7fcd2d5ed266dddb、配置SHA256=68ec5ae8cf77893d5506f807ec201d7adf425e8f2ef8334780503cb8f4a9cfc2。原始report SHA256=6458c0f67455f94fd740c4eed1e02e7348448fb866383ebc6f29e3c87628b249；27项原始一致性审计在修改API前通过。随后仅修非流式/chat用量累计与metadata；正式SSE候选未改，10专项、tests/v2 162项及tests全180项通过。不能把后续交付HEAD说成已经重跑300题。与[用户版](PLAN-FOR-YOU.md)使用同一决定，结果见[RESULTS.md](RESULTS.md)，方法见[METHODOLOGY.md](METHODOLOGY.md)，复现见[REPRODUCTION.md](REPRODUCTION.md)。

原文查看的分工：OpenViking参与实际入库、语义检索及入库回读验收；用户查看原文时，产品接口读取本地Catalog/SQLite的同版本正文并校验hash。每题原文回读不是再次向OpenViking取正文。

## 目标、范围、授权

仓库GuoLai806/GuoLai806-law-ai-advisor；分支feat/versioned-legal-rag-20261105；基线8f433fb2e0dd1ac171c7c797de571c2e8840eb3b。正式生产与runner提交7231a1f；后续API修复身份另记，完整回归180通过。用户授权开发、测试、commit/push公开独立实现和脱敏证据。本轮只改RAG，不改GCode/MyAgent/Pi/企业代码/生产/原简历Word。

目标：真实OpenViking→FTS5/RRF→重排→token预算→云SSE→引用结构检查→原文hash回读→完成答案会话回读，留全部失败、重试与用量，不承诺预设分数。实测检索nDCG10=.794200510、Recall20=.935888889、MRR10=.762533069；标签212/300=.706666667、macroF1=.739557846、doc+labelF1=.535773710、流程269/300。

## 仓库、接口与依赖

- evidence_rag/retrieval.py：真实find，固定scope、level2，共享RankingExecutor；后台完成才释放槽，线程只处理已加载证据。
- viking_index.py：delivery_unknown先落journal，未知结果先reconcile；vectors_only/no_split，L2=scope/unit_id/unit_id.md；内容hash及自身全文可见性检查。
- budget.py：官方固定tokenizer、return_dict=False，窗口32768、framing512、output8192，证据上限4/6000 tokens，完整单元不截断。
- answer.py：Pro enabled/low、8192输出/60秒；仅完全无delta且空响应/超时/无完成标记时同证据同模型disabled重试一次，45秒/2048；部分文本、限制终止、取消不重试。默认不启用；usage缺失不补0。
- api.py：POST /api/v2/chat/stream；GET /api/v2/models runtime_identity；GET /api/v2/units/{id}；GET /api/v2/sessions/{id}/turns；GET /网页。异常仅记录类型、phase及HTTP状态，无上游正文/URL/密钥。
- benchmarks/run_user_flow.py：真实HTTP/SSE，格式无效仍读session；classification INVALID与user_flow_completed分开；正式test要求冻结selection config hash，前后核对源码/运行身份。
- tools/select_user_flow.py冻结最终pair；analyze_user_flow.py做TREC/CI/分位/已知用量；audit_user_flow.py核对raw/journal/report/config/gold/source及前后模型、后端身份。

Python3.12.13/Torch2.15/Transformers5.22.1/ST6.2/MPS float16/M5 Pro48GiB，依赖见pyproject及constraints-models-tested.txt。OpenViking0.5.3独立环境；BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181，1024维；Qwen重排22e683669bc0f0bd69640a1354a6d0aebcfeede5，BGE重排953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e。生成deepseek-v4-pro是云alias，保存实际fingerprint，不声称固定云权重。模型/tokenizer文件hash前后核对。运行中不改source_hashes集合。

## 数据、顺序和验收

1. scifact_full_v3，scope=viking://resources/rag/scifact/scifact_full_v3/。5183资料已导入、回读和自身可见性通过。不重建，不启用L0/L1摘要；复杂PDF/OCR/长文层级检索未验收。
2. 冻结benchmarks/manifests/scifact-validation.json，train32只用于选型；BEIR test300用于检索，同ID/文本SciFact公开dev300用于标签。300题曾用于旧参考后端，不称全项目从未见过。gold/cited_doc_ids不送模型，不改标注。
3. 旧候选与失败全保留。最终仅qwen-final-validation-v2/bge-final-validation-v2，同源码与生成配置，各完整32一次。全记录/source/runtime一致/无已知代码缺陷才进入选择；真实服务失败仍INVALID，不筛零失败。依次最大macroF1、引用文档标签F1、nDCG10、最小用户中位等待；不证明32题稳定优越。
4. selection冻结配置原字节hash、验证报告hash、源码/模型/tokenizer/索引身份后，串行单客户端跑openviking-formal-test-v1全部300；正式结果不调参、不筛题、不挑最好一次。新版本须另协议并保留本轮。
5. TREC9.0.8逐题与总体复核；query bootstrap10000/seed20261106/95%；统一NumPy线性插值p50/p95。分项报告检索、最终证据覆盖、标签、引用文档标签、流程、失败、重试、usage完整性；缺报不能算总费用。
6. 网页实操提问→真实SSE完成→打开原文，另测停止撤回，截图留本机。API/合成测试不能替代实操。
7. 审计raw hash、journal行与report、config/gold/source/runtime，前后模型文件及后端身份。按评审技能由两位独立reviewer检查代码/评分及证据/声明，主agent核实发现；审计工具新增部分记录作者复核的限制，冻结版本wheel-v8保留；后续API修复wheel在v9构建、同字节在v11独立环境安装，依赖复用限制见文末。
8. 双版本最终结果、技术理由、行业比较、简历同步；公开全rows/排名/labels/失败/耗时/usage/hash，移除正文/请求/query/events/凭据/私人路径，登记原始和公开副本SHA。扫描后正常commit/push、核对远端，不force。

## 部署、回退和待确认

本机OV19443、embedding19444、RAG19445，私有凭据经stdin，沿用已授权官方provider，不复制私有配置/helper。服务名称与输出独占，不覆盖旧token/report/journal。

保留旧提交、失败、raw、索引与一致DB备份，只停核对属于本轮的PID。生产目标和容量测试属于后续，本轮无必需用户输入。外部同task/revision检索仅作参照，不称全流程SOTA、所有领域、生产并发或官方隐藏sentence-rationale榜单成绩。

## 正式审计和后续修复的版本边界

正式原始审计发生在非流式API修复前，核对当前冻结source_hashes、raw、journal、report、config、manifest、gold、selection、model before/after与backend before/after，共27项全部通过。后续源码变化使最终HEAD的source_hashes与正式配置不一致是预期现象。重做历史原始审计必须在7231冻结checkout中使用交付版tools，不能在最终HEAD取消source检查。公开report可复算数值，不能替代本机raw与原文审计。

/chat返回usage_attempts、generation_attempts、generation_retries、observed_answer_models、usage_complete、usage_source。每attempt取最后累计usage，跨attempt只累加已报告值；全部尝试都有报告为provider，部分为provider_known_attempts，尝试但无报告为unavailable，未生成为not_generated。legacy无attempt usage归1；仅文本stream仍可回答但usage未知。成功、非法引用拒绝、异常fallback都保留accounting；没有把未知补0。

## 失败、用量和计时验收结论

300原文回读，269会话回读与流程完成；31失败保留：17门槛导致evidence_only、12ConnectError、1RemoteProtocolError、1TimeoutError。10题重试，293生成attempt，283题尝试生成，269题usage可用，259题attempt报告齐全，已知608961 tokens。错误类型不等于根因；price reference只涵盖已报告调用，不是总费用。

原报告逐题user_total_ms线性分位p50=30311.856625/p95=67113.607462；rank_and_rerank p50=16375.5/p95=24458.1845。但UTC跨度25073秒，逐题monotonic和10378.623392秒，pmset实际Sleep交集14756秒。Python3.12.13 mach_absolute_time时钟与跨睡眠日历等待不同；无逐题UTC，不修正或筛掉睡眠期间题。不能据此声称持续唤醒线上延迟或QPS。host-timing-audit提供匿名状态和时钟依据。

UI验收在formal后、API修复前完成3个场景：evidence_only明确降级、真实答案完成且原文可展开、点击停止后草稿撤回与控件恢复。截图仅留本机；停止UI不证明后台计算瞬间结束。

## 下一轮开发顺序（建议，不在本轮正式数据上重新选参数）

1. 为benchmarks/run_user_flow.py增每题UTC start/end和wall_ms，并保留monotonic_ms；保存时钟implementation、主机唤醒条件和运行窗口电源事件。所有请求仍串行留失败，不用时钟差筛题。速度测试启用明确的防睡眠运行包装，只影响本轮进程，结束即释放。
2. 在独立开发数据增加NEI/无法回答、相近但不支持、支持/反驳和多证据切片，保留原标注来源与split hash。使用train切片标定每种reranker阈值，分别报告false release/false refusal、coverage、label macroF1、doc+labelF1、completion、latency；不把分数当概率。候选阈值、数据划分和选择顺序在跑新正式集前冻结。
3. 比较Qwen/BGE、不同证据数量和证据判别器的完整答案对照；同语料/输入/生成模型/输出预算/重试和评分，保留配对逐题数据。先排空已知工程问题，再冻结质量优先与延迟门槛；不存在本轮已证明Qwen稳定最优的结论。
4. 新增至少一个中文和一个跨领域通用任务，以及长文与无法回答问题。新的正式集不得用作这次开发选择；若公开数据可能已见，应明确它是复现评测而非严格未见测试。人工语义引用支持与判别一致率分开；缺少人工标准时不冒称高精度。
5. 在持续唤醒且指定硬件、缓存、并发条件下测完整用户wall latency，再做容量测试；同条件强基线覆盖生成、引用和失败，才评估完整系统的前沿声明。任何改进另run_id和source/config/model身份，保留本轮300和历史失败。

新轮部署以独立venv、端口、DB与scope进行；先probe、索引验证、开发集，再冻结正式集，避免覆盖已验收原始记录。回退用旧源码和保留索引/SQLite备份；模型或schema变化先评估兼容，不直接激活。生产目标、专家标签与并发负载未定，不是本轮等待用户确认的阻塞项。

## 最终构建与安装身份

最终交付源码提交c46cd3b5c2d8695c04b4c22ee4e8eaf8f096b151，正式冻结提交仍为7231a1f。wheel在v9构建，SHA256=04d7652444716a4a4ce4f7c4a44f193a9894b970e454b0813c276965b0b3f758；同字节wheel在v11独立环境安装，pip check通过，网页HTTP200、HTML/API hash与交付源码一致。v11复用v8已独立安装的依赖，逐文件核对2856个非缓存文件；不是全新联网依赖解析。v9下载超时/持续停滞和v10环境创建失败全部保留，不算RAG质量失败，不删除。入口probe未加载模型或重新跑300题。

[安装入口证据](evidence/wheel-clean-entry-v11.json)、[依赖来源与文件核对](evidence/installed-dependency-provenance-v11.json)、[代码和完整测试](evidence/delivery-validation-v1.json)分别保留。
