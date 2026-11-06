# 真实 OpenViking 30 题前后对照：结果、等待和限制

两组已经完整结束；每组30题、失败仍在分母。同一被测实现提交 `561ec65`、同一OpenViking0.5.3索引和同一题目，旧配置21/30、新配置22/30。新方案的引用更集中、NEI多答对2题、完整流程多2题；总体正确率变化小且区间跨0，反驳类退步、等待和已报用量增加。保留候选为可选模式，默认继续text。

[用户版](PLAN-FOR-YOU.md)讲使用和取舍；[AI版](PLAN-FOR-AI.md)记录接口和冻结协议；[选型](TECHNOLOGY-DECISIONS.md)、[复现](REPRODUCTION.md)、[验收历史](VALIDATION-HISTORY.md)对应同一版本。

## 范围与实际系统

正式30题各类10，train开发6各类2，按seed20261107固定抽样。SciFact公开且历史已涉及，不是隐藏未见集。BEIR5,183篇完整摘要，真实OpenViking find + BM25 + BGE-M3 + RRF + Qwen3-Reranker-4B；scope=scifact_full_v3，实际云deepseek-v4-pro思考enabled/low。模型、tokenizer、source/config/manifest/selection/raw摘要全部保存。warm旧索引、Apple M5 Pro/MPS float16、单用户串行、caffeinate；baseline先跑，再candidate。

旧配置text+reranker_threshold0；新配置grounded_json+grounded_evidence。其余模型主参数/召回/证据预算相同，但JSON附加提示与gate一起改变，不能把变化归因单因素。全部60题检索排序相同，预算后最终证据平均Recall均.833333；检索前20高覆盖不等于答案收到的证据完整。

## 质量与失败

| 指标 | baseline | candidate |
| --- | ---: | ---: |
| 三类标签一致率，全部30 | 21/30 = .700000 | 22/30 = .733333 |
| macroF1 | .713247 | .735779 |
| SUPPORT正确 | 10/10 | 10/10 |
| CONTRADICT正确 | 7/10 | 6/10 |
| NOT_ENOUGH_INFO正确 / F1 | 4/10 / .533333 | 6/10 / .666667 |
| citation文档+标签precision | 16/37 = .432432 | 15/22 = .681818 |
| citation文档+标签recall | 16/20 = .800000 | 15/20 = .750000 |
| citation文档+标签F1 | .561404 | .714286 |
| 全流程完成 | 27/30 = 90.00% | 29/30 = 96.67% |
| 原文回读 / 完成会话回读 | 30/30 / 27/30 | 30/30 / 29/30 |
| 失败 | 3 gate evidence_only，未生成可交付答案 | 1 invalid_quote，GroundingRejected |
| 检索nDCG@10 / Recall@20 / MRR@10 | .864658 / .966667 / .832593 | 完全相同 |

candidate失败qid879，标准CONTRADICT，错误摘录硬拒绝，不修JSON或改判NEI。baseline3门槛失败全属标准NEI，不能补为正确NEI。两个runner返回1是包含失败的完成报告，30行/journal齐全；未重跑。

citation指标是本项目微平均“文档ID、结论标签一起命中”，只有预测SUPPORT/CONTRADICT计支持引用；NEI参考摘录不计支持。失败没有预测引用，gold仍保留。候选减少了预测引用，正确对数却从16降到15：F1变好主要是引用更精确，并非覆盖全部提高。不是SciFact官方句子rationale榜单，也不是逐句人工语义支持。

29次合法grounded交付：28有摘录、1个NEI无摘录，合计核对29个span。摘录检查保证空白规范化后连续出现在所选标题/正文、ID和结构合法，不保证摘录能推出结论。不能用29/29字面校验包装“语义支持100%”或“零幻觉”。

### 按同题比较与不确定性

配对19题都正确、6题都不正确、3题改善、2题退步。总体差+3.333个百分点；10,000次paired query bootstrap、seed20261107、95%区间[-10.000,+16.667]百分点，exact McNemar两侧p=1。单组Wilson区间：baseline[52.124%,83.335%]，candidate[55.552%,85.817%]。

补充指标配对重采样：引用precision差+.249386，95%[.066667,.414103]；recall差-.05，95%[-.166667,0]；F1差+.152882，95%[0,.275546]；NEI准确率差+.2，95%[-.2,.6]，NEI3改善1退步。引用F1区间碰0，NEI跨0；不能宣称稳定总体领先。附加指标是事后解释，未据此改模型或选配置；共享文档问题可能相关、单次云生成波动未测、多指标未校正，不能把一个区间作为普遍显著性证明。

[paired](evidence/paired30-v1.json)保留逐题迁移；[附加区间](evidence/paired-additional-v1.json)与[独立复算脚本](evidence/recompute-paired-additional.py)可在公开数字上重算。全部失败仍在分母。

## 实际日历等待与用量

以下统一NumPy线性插值，与paired工具一致。完整等待包含SSE全部结束、原文回读和会话回读；未完成题仍计等待。首字只含有可见正文的题，思考token不算可见答案。

| 秒、样本数 | baseline p50 / p95 | candidate p50 / p95 |
| --- | --- | --- |
| 全部30完整用户等待 | 25.911 / 38.002 | 29.396 / 52.712 |
| 仅完成流程，27 / 29题 | 26.173 / 38.755 | 28.772 / 53.051 |
| 首个可见正文，27 / 29题 | 25.287 / 37.613 | 28.760 / 53.034 |
| 全部30 SSE结束 | 25.894 / 37.983 | 29.383 / 52.696 |
| 整轮日历跨度 | 849.187 | 923.889 |

runner原始order-statistic分位数同时保留，数值不同是算法不同，不能混进这张前后表。60题全部wall有效，0倒退、0大于1秒双时钟差异flag；原始UTC/ns与事件独立重建审计通过。历史300曾跨睡眠，不能将旧秒表延迟拿来声称本轮提速。两组串行，缓存命中、云生成与共享桌面条件不同，不能推断延迟因果或生产SLA；本轮观察到候选等待更长。

| 已报告用量 | baseline | candidate |
| --- | ---: | ---: |
| prompt / completion / total tokens | 45,490 / 14,237 / 59,727 | 62,624 / 26,697 / 89,321 |
| 其中reasoning tokens，已含completion | 11,506 | 22,873 |
| prompt cache hit / miss | 41,344 / 4,146 | 19,200 / 43,424 |
| 生成attempt / 重试题数 | 28 / 1 | 30 / 0 |
| usage可用题 / attempt全量用量齐全题 | 27 / 26 | 30 / 30 |

baseline3题无生成，另1题重试中的一个attempt缺报；未知不能填0。candidate失败attempt也报用量、全部30完整。不能把已知token或按价格算的参考费用写成实际总账单，更不能写成本下降。

## 工程与正常使用验收

完整回归310 passed/5warnings；两独立review分别144项及7个边界、另一范围76项通过，测试存在重复不累加唯一数。源码与冻结config/模型文件字节未改，两正式材料audit全部通过，TREC v9.0.8逐题和总指标在输出四位小数舍入容差内一致。

backend raw前后只有after额外含embedding.load_seconds诊断字段；显式身份字段、OV版本/config/scope、embedding model/revision/device/dtype/torch一致。完整快照和独立身份投影同时保留，未冒称原样字节相等。

wheel0.2.1 SHA256 `cbdcff20aca3c70300fa2207fbe3fbd4dc9d2579e0c688edfb597ae93c532376`，独立安装环境的模块和HTML字节与源码相同，CLI入口通过；显式.pth复用本机运行/模型依赖。完整联网依赖下载慢导致两次终止和早期检查失败均保留，未验任意全新机器从零安装。

安装后19447实际API：固定开发qid1134，SSE可读答案、verbatim核验、真实OpenViking trace、4条同版本正文hash、完成会话回读全通过。服务刚启动的一次ready观察503(answer_model_health)保留；之后API验收和最后ready为成功，未把startup未ready算验收通过。浏览器验证同开发题正常回答、查看原文、一次pre-completion停止撤回通过；源码来源及HTML hash一致。UI/API smoke不进30题质量/延迟统计。

[安装记录](evidence/installed-user-flow-v1.json)、[安装身份](evidence/installed-identity-v3.json)、[网页摘要](evidence/ui-validation-v1.json)保留验收状态和原始截图/文本hash，截图正文留本机。生产、并发、中文/法律人工语义、长文L0/L1/OCR未验。

## 保存与版本索引

所有原始报告、journal、答案、事件、失败详情、配置、数据源、模型字节清单保留本机。公开5报告78行包含dev6-v1/v2/v3及正式两组；配置原字节不变、题ID/gold/排名/失败/usage/时钟/hash完整，递归去掉正文与私人路径。公开文件含original_report_sha256，与其自身public hash不同；[脱敏审计](evidence/PUBLIC-REDACTION-AUDIT.json)明确两者。

公开selection的路径已脱敏，pair hash对应原本机冻结材料；新机器须按复现说明重建并重新冻结本机输出路径。公开数值可复算，不等于公开了全部原始SSE或人工语义证据；完整原始事件审计在本机完成。

manifest SHA256 `8ee8fd71d89777b93ef27e062fffd81ddb8f419b7c476eddc16bf2a08c6d4690`；原baseline report SHA256 `92698b53175bfe51724fa29652439e3a77dad4b7379d95803970a201aaefb8d2`；原candidate `80dd67a802f0acf1477dd476b30f95db93512a635bd55ad98d3a032e569c1911`。源码/config/后端身份见两配置、tested-source、model/backend前后清单。

旧300真实结果保留在[历史结果](../openviking-user-flow-20261106/RESULTS.md)。历史nDCG=.7942、Recall@20=.9359，当前均衡30的.8647/.9667不构成检索提升，两样本不同。当前数据只能支撑本次固定公开小样本上的具体变化，不能证明“完整通用RAG前沿”。
