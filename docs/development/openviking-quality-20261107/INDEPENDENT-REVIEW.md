# OpenViking 质量轮最终独立交付复审

**结论：未发现需要修复的确定缺陷。** 当前固定被测源码、完整真实评测、公开导出和权威交付文档符合本轮约定范围，可以交还主任务完成归档、副本同步与普通推送。工程验收和记录一致性通过；本轮没有证明稳定行业前沿或通用问答泛化。远端最终文档提交与运行服务的最后状态由主任务确认，本审查没有代为推送或操作服务。

审查对象是实现提交 561ec65292218541b380ff3963d21fb6115e6778 的正式源字节，及 repo/docs/development/openviking-quality-20261107 下最终两 PLAN、RESULTS、TECHNOLOGY-DECISIONS、RESUME-REWRITE、VALIDATION-HISTORY、REPRODUCTION、evidence/README 和公开证据。精确被审内容摘要在 [FINAL-REVIEW-HASHES.json](evidence/review/FINAL-REVIEW-HASHES.json)。原始 source/config/report/journal/selection 均未修改。此前两 P2 修复与 144 回归/7 模拟边界见 上一阶段复审（原始报告本机保留），不重复累加测试数。

## 角度一：需求与交付覆盖

| 需求 | 实际覆盖与结论 |
| --- | --- |
| 必须实际使用 OpenViking | 两组正式共 60 个 raw SSE 的 dense_backend/scope 与冻结配置一致；每组 30 完整行。沿用历史核验的 5,183 篇真实 L2 索引，未把本轮说成重新导入或 L0/L1/OCR 实测 |
| 全用户路径 benchmark | 实际 POST/SSE、交付、同版本原文与配置要求的会话回读；27/30 与 29/30 完成单列。科学判定与流程完成分开，全部失败仍在质量分母 |
| 约十分之一题量与冻结 | train6 每类2，test30 每类10，固定 seed20261107；完整 dev6-v3、双配置与 selection/pair/正式路径绑定，先冻结后发题；两正式各一次 |
| 选型有理由或数据 | 12 项选型区分用户指定、历史同题检索证据、质量优先规则、工程常量与未知。没有把未做参数寻优、强系统对照、并发或多领域验证说成实证最优 |
| 保存详细测试与失败 | 5 份报告 78 行，含 dev6 v1/v2/v3 和正式两组；历史失败与原始 hash 保留。310 项全回归 XML 零失败，独立范围与 mock 不代替真实系统质量 |
| 安装后实际体验 | wheel 包内源码与 HTML 全等于 repo；api/answer/grounding 实际安装路径与 hash 一致。安装 API 原始 SSE/原文检查与保存的 UI 正常问答、原文面板、停止撤回、截图/AX hash 一致 |
| 大白话与开发依据双版 | 用户版独立解释结论、现状、动作、依据、损失、使用和未验范围；AI版明确文件、契约、数据、配置、顺序、测试、回退和授权。两者使用同一组决定 |
| 简历能经追问 | 推荐工程实现和历史300题检索指标，30题量化句保留公开均衡样本范围；明确整体仅多1题、反驳退步、CI与等待代价。未把私人简历或企业/微调旧数字借作本轮成果 |
| 公开独立实现与资料 | 实际公开五报告、配置和 journal 与原始数据脱敏投影完全一致，78 行失败和数字均保留；没有 raw SSE、问题/回答/来源正文、私人文件、实际模型凭据或本机路径 |

双版及全部结果、选型、简历、验证、复现与 evidence 索引的 66 个本地 Markdown 链接都存在；复现命令引用的原 manifest 存在。权威只认 repo 目录，根目录/outputs 副本由主任务接手同步。文档没有预先冒称最终远端资料同步已成功。

## 角度二：数据可信度与独立重算

本审查脚本不导入仓库 scorer/audit/runner 辅助函数，从保存的原始 SSE、gold、qrels、UTC/ns 与 journal 独立提取和计算。五报告的准确率、macroF1、NEI、引用文档+标签 precision/recall/F1、每题检索排名、失败数、已知 attempt usage、真实 wall 与首个正文、原始文件摘要均与保存值一致。dev6 三配置除 source_sha256 外完全相同，v1/v2/v3 都 4/6，v2 一个摘录错误仍留在记录；只 v3 绑定正式冻结。

| 正式指标，全部失败保留 | baseline | candidate |
| --- | ---: | ---: |
| 标签正确 | 21/30 | 22/30 |
| SUPPORT / CONTRADICT / NEI 正确 | 10 / 7 / 4 | 10 / 6 / 6 |
| macroF1 | .7132468853 | .7357788410 |
| 文档+标签 correct / predicted / gold | 16 / 37 / 20 | 15 / 22 / 20 |
| 文档+标签 precision / recall / F1 | .432432 / .800000 / .561404 | .681818 / .750000 / .714286 |
| 用户流程完成 | 27/30 | 29/30 |
| 已知 total tokens / 用量完整题 | 59,727 / 26 | 89,321 / 30 |
| 全30 wall 等待 p50 / p95，秒 | 25.910801 / 38.002353 | 29.395997 / 52.711807 |

等待前后表统一 NumPy 线性插值，报告中的旧离散分位原值同时保留。完整等待含源与会话回读，失败等待留在 all 行；首个正文只有实际正文题进入样本。正式60题 wall 有效、无回退及超过1秒双时钟差异 flag，UTC/ns 重新计算一致。原始单调时钟与日历时钟不混用，历史跨睡眠300题延迟没有拿来作本轮提速证据。

配对独立复算为 19 都正确、6 都错误、3 改善、2 退步；accuracy 差 +3.333pp、10,000次 paired bootstrap95%[-10,+16.667]pp、exact McNemar p=1；单组 Wilson 区间均与保存值一致。附加引用与 NEI 的独立重采样，以及公开纯统计脚本，均复现原结果：citation F1 差 +.152882，95%[0,.275546]；NEI 差 +.2，95%[-.2,.6]。附加指标事后解释、多指标未校正、共享文档可能相关，不支持稳定总体优胜。

candidate29次合法 validation：28次有摘录、1次NEI无摘录，共29个 span。所有 span 的引用ID、字符数、hash、空白规范化 title+text 连续出现都独立通过；仍没有做人工语义支持判断。candidate1个 invalid_quote 硬拒绝，baseline3个 gate evidence_only，均不改判NEI、不筛分母。

模型文件清单的 models 字段前后一致；后端 raw after 只新增 load_seconds 诊断字段，去掉该新增字段后其余身份完全一致。文档明确“身份不变”而不是“快照原样字节相等”。云 alias/fingerprint 不代表固定权重。baseline一次重试中缺报 attempt usage，未补0或声称完整账单/成本下降。

公开 source/config/hash、报告及 journal 全字段投影核对通过，PUBLIC-REDACTION-AUDIT 中原始与公开摘要都与实际文件一致；最终目录私密路径、credential-like、私人简历文件引用扫描未检出。演示 loopback-demo 是本机公开演示令牌，不是上游模型凭据；截图和原文正文保留本机。公开 selection 路径脱敏后不能重放原 pair，复现须重新冻结本机路径，文档已明确。

## 证据位置与审查限制

- [全部原始记录独立复算](evidence/review/all-recomputed-v1.json)、[配对复算](evidence/review/paired-independent-v1.json)、[引用字面核验](evidence/review/quote-independent-v1.json)。
- [附加区间独立重算](evidence/review/additional-paired-independent-v1.json)、[公开脚本输出](evidence/review/published-additional-stdout-v1.json)。
- [实际公开材料核对](evidence/review/public-evidence-independent-v1.json)、[导出独立探针](evidence/review/export-probe-v1.summary.json)、[文档链接](evidence/review/docs-links-v1.json)。
- [wheel/身份/冻结复核](evidence/review/identity-freeze-recomputed-v1.json)、[安装/UI原始证据复核](evidence/review/functional-evidence-independent-v2.json)、[开发配置一致性](evidence/review/development-config-independent-v1.json)。

identity-freeze 首版直接比较后端 JSON 得到 false 的原因是仅有 load_seconds 新字段；paired 独立复算中的显式身份投影为 true。UI 首版探针错误假设原文面板重复显示 E1，该探针失败保留；第二版以真实面板契约核对引用原文标题和正文，全通过。这两项是独立探针口径纠正，未修改产品、正式报告或原始证据。

本审查读取保存的原始安装 SSE、UI accessibility 文本与截图 hash，没有重新操作浏览器或判定每个答案语义正确。原文与会话 GET 结果以 runner 的 hash/回读记录和独立功能验收为依据，未伪称保存了每次 GET 的完整 HTTP 响应。旧 audit 核心曾由本 agent 编写，存在作者自查限制；本轮主要新实现作者为其他 agent。没有调用真实模型、重跑正式测试、操作 Git/服务或修改任何 repo 文件。

## 明确停止状态

最终被审文件摘要已固定。独立复审报告和当前任务双版说明已保存；**本子任务写入已停止**。后续归档、最终公开清单、新增交付记录、副本同步、普通提交/推送和远端确认交给主任务。不把尚未完成的远端操作或生产验收计作已完成。
