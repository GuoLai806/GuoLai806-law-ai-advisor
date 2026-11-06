# OpenViking 回答质量与 30 题对照：最终执行及交接依据

[用户版](PLAN-FOR-YOU.md)与本文同一组决定。仓库 `GuoLai806/GuoLai806-law-ai-advisor`，既有分支 `feat/versioned-legal-rag-20261105`；本轮实现固定提交 `561ec65292218541b380ff3963d21fb6115e6778`，基线 `b62887d686894f50811ba54d0c4ed08f4b6e8fc9`。正式源文件摘要由两份 config 和 [tested-source](evidence/tested-source-v1.json)绑定；最终文档提交不同于被测实现提交，不改变测试源字节。

## 目标、结果和不可改变的边界

用户明确要求保留实际 OpenViking、面向通用 RAG、详细原始测试、选型有理由或数据、改善资料不足判别和引用、补准全流程计时；正式题量从历史 300 缩至 30。已授权普通开发、测试、commit/push；gcode 和本 RAG 保持公开，只推可公开独立实现。该轮仅 RAG，GCode/MyAgent/Pi/企业实现/原 Word/生产配置未改。

| 要求 | 本轮实现与验收 | 结果证据 |
| --- | --- | --- |
| R1 真实 OpenViking | 60 次正式用户 SSE 请求实际 dense find，锁定 scope；每题版本原文和会话回读 | 两报告/两 audit；安装 API 检查 |
| R2 约十分之一 | train 每类 2 的开发 6；test 每类 10 的正式 30；baseline/candidate 各一次 | manifest、双 selection、全部 journal |
| R3 提高 NEI 与引用 | 可选 grounded_json + grounded_evidence；一次正常生成、摘录字面核验、失败硬拒绝 | NEI4→6/10，citation F1 .5614→.7143，整体21→22/30；反驳7→6/10，统计不支持稳定优胜 |
| R4 真实等待 | UTC/ns、wall/monotonic、逐事件快照、独立审计重建；caffeinate | 60 题时钟全部有效、0 flag；同算法全流程 p50/p95 |
| R5 保存证据与理由 | 5 份完整报告 78 行、全部失败/usage/版本；12 项选型说明 | evidence、RESULTS、TECHNOLOGY-DECISIONS |
| R6 正常使用与长期归档 | 310 回归、两角度复审、wheel/CLI/安装 API/网页原文与停止；双版同提交普通推送 | VALIDATION-HISTORY、安装 identity、UI 摘要、GitHub 交付核对 |

本轮已经达到实现和约定小样本验收范围。高精度长期目标仍有缺口，不能把小样本验收成功写成“稳定业界前沿”。不改历史题目、gold、索引、模型权重和报告；不合并 PR、不 force、不改仓库可见性、不发布生产服务。

## 代码、接口和决定

- `evidence_rag/grounding.py`：严格解析 `{decision,answer,evidence:[{id,quote}]}`；decision 支持 SUPPORT/CONTRADICT/NOT_ENOUGH_INFO/ANSWER。拒绝重复 JSON key、未知或重复 ID、额外字段、非法字段类型、正文引用不一致、虚构摘录。仅空白规范化 title+text 的连续 substring；12–1200 字符、至多4条；SUPPORT/CONTRADICT/ANSWER 至少一条。NEI 可零摘录，参考来源不冒充支持证据。该检查不独立证明语义支持。
- `answer.py`：`OpenAIStream.answer_format=text|grounded_json`，默认 text；`messages_for`与实际预算同契约。grounded 缓冲上游 SSE 正文至多64KiB，先转发身份/attempt/usage，正常终止后输出 grounding_validation，再单个可读 answer_delta；不伪造逐字动画。非空格式错误不修 JSON、不重问模型。仅旧规则符合条件的完全空输出最多重试一次，记录所有尝试。
- `api.py`及`benchmarks/serve_user_flow.py`：operator `ANSWER_FORMAT`、`GENERATION_GATE`。默认 `text + reranker_threshold(0)`；可选 `grounded_json + grounded_evidence`仅 general 生效，允许低 rerank logit 输入判断，仍要求 ready、非降级、完整证据和可验证预算。legal 原限制不放宽。JSON/SSE 保留校验、失败阶段与用量；runtime identity 含格式、gate、prompt hash。
- `static/index.html`：用户只看正常正文、引用和原文，不看内部JSON；停止撤回尚未完成正文；完成引用提示明确只做原文摘录核对。
- `benchmarks/run_user_flow.py`：新 manifest 的 short_evaluation 绑定固定30；没有该字段的历史manifest仍保持300。每题保存 UTC/ns、双时钟、原始事件 wall；source/session 回读失败仍保留 grounding 和已知usage。科学评分 grounded 要 exactly1 validation、三科学标签且正文首行一致；ANSWER 可在产品完成，但科学评分记 INVALID。
- `tools/prepare_user_flow_subset.py`：以 seed20261107 和 query_id SHA256 每类固定抽样，不覆盖。`tools/freeze_short_run.py`、`benchmarks/short_protocol.py`：正式发题前绑定完整 development6 report/journal、双selection互指和共同摘要、manifest/source/config/runtime/输出路径；禁止只有两hash绕过，正式输出或journal存在禁止重新冻结和重放。
- `tools/audit_user_flow.py`与`audit_flow_timing.py`：原始SSE、gold、源码身份、标签/引用/attempt usage、UTC/ns、缺失/回退逐事件重建；不复用runner计时公式掩盖错误；异常保留质量分母。`analyze_user_flow.py --trec`独立TREC复核排序；`compare_short_user_flow.py`失败包含的同30配对、Wilson、paired bootstrap、exact McNemar及统一NumPy wall分位数。额外引用与NEI区间在evidence中有独立复算脚本。

## 依赖、数据和正式配置

Python3.12/macOS arm64，Apple M5 Pro、48GiB共享内存、MPS float16、Qwen batch8。OpenViking0.5.3、5,183篇已逐篇核验的 L2、scope `viking://resources/rag/scifact/scifact_full_v3/`。复用已有真实索引，未本轮重新导入；产品原文从 Catalog/SQLite 同版本读，OpenViking 内容回读发生在入库验收。L0/L1、OCR、多模态、层级长文未验收。

BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`；Qwen3-Reranker-4B `22e683669bc0f0bd69640a1354a6d0aebcfeede5`。`deepseek-v4-pro`云别名，thinking enabled/low、temperature0、8192输出、60秒；符合规则的空输出一次 disabled/45秒/2048补救。上下文32768、framing512、output reserve8192、最多4完整单元/6000证据token。云alias/fingerprint不是固定云权重，不承诺逐字重现。

`benchmarks/manifests/scifact-short30-20261107.json` SHA256 `8ee8fd71d89777b93ef27e062fffd81ddb8f419b7c476eddc16bf2a08c6d4690`。开发来源 train；正式使用 claims_dev 与 BEIR test。三类均衡、公开历史已见，不是隐藏未见集。baseline config SHA256 `a577eb68436a64762e26f567334a8922ac810fc6950e883b7768768a77f83ea7`；candidate `577bc60feaee1a89da84dbda987f788f11aab2edc2d8b843464cff788d928664`。两者除已声明的联合格式/提示/gate之外相同；全部检索排名和预算后证据覆盖也一致。不能把收益归因单因素。

## 已执行顺序与验收

1. 记录双版、按互斥范围实现、回归与两角度独立review；修科学ANSWER计分、失败丢metadata、冻结绕过、计时审计问题。
2. 保存固定6/30 manifest。开发v1/v2是修评分与元数据bug之前；v3是最终源码复核，6/6流程完成、4/6正确，audit通过。模型/提示/参数未按正式题调整；全部早期记录保留，只v3绑定正式冻结。
3. 同一固定最终源码下发题前冻结baseline与candidate，串行各跑30一次。runner因保留失败返回1并保存完整report；这不是重跑理由。两审计和TREC均通过，所有失败保留。
4. 对全部30做同题比较、计时与usage审计、模型文件和后端身份前后核验。raw after比before多load_seconds诊断字段，完整快照保留；身份字段比较一致，不能称快照原样字节一样。
5. 停止核实属于本轮的正式19446服务，再启动独立安装wheel的19447入口，不让额外模型实例影响正式延迟。安装API完整SSE/真实OV/4条原文hash/会话回读通过；浏览器固定开发题正常答案、查看原文、pre-completion停止撤回通过，原始截图与AX留本机。功能smoke排除30题质量统计。
6. 导出5报告78行，数值、排序、失败、usage、hash不变，原题正文/答案/论文/SSE/密钥/私有helper/权重不推。双版、结果/选型/简历/复现/验收历史与README入口同提交；最终独立review、扫描、普通push，并读取远端commit及文件内容确认。

## 当前结论、统计和使用入口

21→22/30；NEI4→6/10；citation F1 .561404→.714286（precision .432432→.681818、recall .8→.75）；流程27→29/30。配对3改善2退步，accuracy+3.333pp，bootstrap95%[-10,+16.667]pp，McNemar p1。引用F1差区间[0,.275546]，NEI差区间[-.2,.6]，不宣称稳定改善。candidate1次invalid_quote硬拒绝；baseline3次gate evidence_only。29合法grounded交付中28有摘录、1无摘录，29摘录span只保证字面存在。

wall完整等待全30、同NumPy线性插值：baseline p50/p95=25.911/38.002秒，candidate29.396/52.712；首个正文27/29样本p50=25.287/28.760秒。全部60wall有效，0倒退/大于1秒双时钟差flag。串行缓存/网络/共享桌面未随机化，不宣称速度因果或生产SLA。已知tokens59727→89321，baseline26/27生成题attempt用量完整、candidate30/30；缺报未知，成本无完整账单。

wheel版本0.2.1、SHA256 `cbdcff20aca3c70300fa2207fbe3fbd4dc9d2579e0c688edfb597ae93c532376`，独立安装环境明确`.pth`复用已有运行/模型依赖，CLI及模块/HTML字节通过。两次全依赖联网下载因速度终止、早期检查失败保留，未把它们算安装成功。未验从零新机器依赖下载。

本机19443/19444为实际OV/embedding；19447为安装后的候选问答，只绑定loopback，演示token `loopback-demo`，本轮保留网页入口；正式19446已停止。私有DB/模型key/auth通过本机安全输入，部署不能把它们写入公开config或shell参数；不读取或修改用户Pi设置。后续机器按[复现说明](REPRODUCTION.md)重建私有状态，不依赖本机绝对路径。

## 回退、后续与授权边界

默认text仍保留，operator切回`text + reranker_threshold`即可回退，不重建索引或schema。历史b62887d与300原始材料、当前Catalog一致备份保留。新机器复现必须重新冻结新路径的pair，脱敏selection保留原pair摘要，不可拿公开副本直接绕过正式检查。

无待用户重新授权项；源码和普通文档推送已授权。生产目标、全新机依赖安装、多领域/中文/长文、人工语义支持和并发压测属于后续。下一轮先在开发集改善反驳/NEI和预算后证据覆盖、比较较轻重排，再冻结新协议；不得覆写本轮成绩或用正式错题调参后声称未见测试。完整复现与验收边界见RESULTS/VALIDATION-HISTORY；该轮结束后不要再次运行相同名称的正式runner。
