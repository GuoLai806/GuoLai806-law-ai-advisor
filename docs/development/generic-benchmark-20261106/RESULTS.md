# SciFact 官方测试集：产品共享排序结果

在预先选定的 **BGE-M3 嵌入 + Qwen3-Reranker-4B 重排**配置下，官方 SciFact test 的 **300/300 个查询全部处理，执行故障 0**。产品共享排序核心在证据预算前的结果为 **nDCG@10 0.794175、Recall@20 0.935889、MRR@10 0.762501**。相同语料、同一测试集上的产品词法基线分别为 **0.668854、0.836556、0.635099**。完整数值和哈希见 [机器可读摘要](scifact-official-test-summary.json)。

这说明该固定配置在这份英文科学文献检索基准上，预算前排序高于此产品词法基线；它**不证明**问答正确、引用支持、法律适用、其他领域泛化，或完整 OpenViking 索引的检索表现。密集侧使用完整 5,183 文档上的本地精确余弦参考，**不是**通过 OpenViking 对完整 SciFact 语料进行查询。实际产品最终还会选择段落、限制证据数量和预算，这里没有计算该阶段的指标。独立的少量文档 OpenViking/HTTP 测试仅支持其实际测试范围。

## 固定选择与范围

选择发生在官方 test 运行之前：先用从官方 train 按查询 ID 哈希选出的 32 条验证查询比较四种组合，再因质量优先选定 BGE 嵌入与 Qwen 重排；轻量 BGE+BGE 是验证集中测得的低延迟备选。两者验证集的 nDCG 差值为 +0.03148，按题配对 bootstrap 95% 百分位区间 `[-0.00197, 0.07910]`；小样本区间跨零，不能说提升已被统计上确定。修复产品证据段选择后，已用全部 5,183 文档逐篇比较输入、重跑选定配置的 32 题验证，并在增加逐题持久化后再次重跑；选定配置和词法基线各由自己的完整验证报告生成固定 selection。**官方测试集结果没有用于重新选择或调参。**

冻结配置 SHA-256：`1c92761be46dc6fac96aad7d3c4fdb9c9d5585d73a5d113dd18c680c5f76a278`。产品路径使用 `Catalog.stage_documents`、产品词法检索、`document_embedding_text`、共享 `rank_candidates` 及产品 `CrossEncoderReranker`；每路候选 top 40、融合后上限 60。嵌入输入为标题加正文，没有注入相关性标签、答案或文档 ID。词法基线没有加载密集模型或重排器；本次 300 个查询中两条路径均没有命中通用标题精确匹配。

## 300 题结果和配对差异

| 系统 | nDCG@10 | Recall@20（逐题宏平均） | MRR@10 | 相关文档 top-20 命中数 | 至少漏掉一个相关文档的查询 |
|---|---:|---:|---:|---:|---:|
| 产品词法基线 | 0.668854 | 0.836556 | 0.635099 | 280 / 339 | 55 / 300 |
| BGE + Qwen | 0.794175 | 0.935889 | 0.762501 | 316 / 339 | 23 / 300 |

Recall@20 是先对每题计算召回再平均，因此不等于表中的相关文档命中数相除。所有排名、原始路分数、每题失败状态和耗时都保留在工作区输出；报告只含指标和索引，不复制原始语料或权重。

用固定 seed `20261106` 对**同一 300 个查询配对**有放回重采样 20,000 次，计算“选定模型减产品词法”的百分位 95% 区间：

| 指标差值 | 平均差 | 95% 区间 | 改善 / 下降 / 相同的题数 |
|---|---:|---:|---:|
| nDCG@10 | +0.125321 | [0.091849, 0.159165] | 105 / 25 / 170 |
| Recall@20 | +0.099333 | [0.066000, 0.134667] | 35 / 1 / 264 |
| MRR@10 | +0.127402 | [0.089329, 0.165722] | 97 / 25 / 178 |

这些区间描述这一批基准查询的配对差异，未作多重比较校正；不能扩展为所有业务场景的质量保证。正式 test 上没有运行 BGE+BGE，所以不能用该表声称 Qwen 重排在 300 题上优于 BGE 重排。

## 执行成本和数据完整性

选定配置使用 MPS、float16、batch 2；记录的实际嵌入和重排设备均为 `mps:0`。本次测试对完整 5,183 篇和 300 查询计算向量，测试集查询对应的缓存未命中并实际重新计算：全库嵌入约 109.74 秒、300 查询批量嵌入约 2.13 秒；完整运行约 5,079.87 秒。**排名阶段**每题 p50 约 16.05 秒、p95 约 21.82 秒；词法基线排名阶段 p50 约 9.73 毫秒、p95 约 13.32 毫秒。两者均不包含完整应用请求、模型加载、建索引、查询编码或答案生成，因此不是端到端延迟。测试期间模型按嵌入模型释放后再加载重排模型，记录的预热分别约 128.55 毫秒和 316.88 毫秒。

两份 test 报告的 journal 均包含固定输入/源码/运行 ID、实际 runtime、300 条依序写入并 `fsync` 的逐题记录和最终报告 SHA-256 完成标记；逐题内容、数量、ID、失败数及最终报告均经独立程序核对一致。测试启动前后的冻结源文件哈希一致。报告和 journal 的 SHA-256、模型 revision、计时及检查结果在 [摘要 JSON](scifact-official-test-summary.json)；逐题原始报告、journal、source-before/after 和 bootstrap 分别保存在任务输出目录。

对两组结果，用 [NIST trec_eval v9.0.8](https://github.com/usnistgov/trec_eval/tree/v9.0.8) 的 `-c -q` 逐题和总体复核 `ndcg_cut.10`、`recall.20`，并对前 10 结果复核 `recip_rank`。所有 300 题和总体指标都在其四位小数输出的舍入误差范围内；最大逐题差为 `0.00004972`。为保留产品自己的排序，传给 trec_eval 的分数按最终名次构造，并在计算前按 BEIR 规则排除查询与文档 ID 相同的结果。

官方来源、许可限制、验证集冻结和先前失败记录见 [基准方法](README.md)。SciFact 测试数据是英语科学资料；NFCorpus 的原始发布方另有用途限制，不能用本次结果推断 NFCorpus、中文或商业授权。

## 原始排名和核验文件

- 正式选定模型：[300 题原始报告](evidence-final/scifact-product-v4-bge-qwen-test.json)、[逐题 journal](evidence-final/scifact-product-v4-bge-qwen-test.json.journal.jsonl)、[完整性核验](evidence-final/scifact-product-v4-bge-qwen-test-journal-verification.json)。
- 正式词法基线：[300 题原始报告](evidence-final/scifact-product-v4-lexical-test.json)、[逐题 journal](evidence-final/scifact-product-v4-lexical-test.json.journal.jsonl)、[完整性核验](evidence-final/scifact-product-v4-lexical-test-journal-verification.json)。
- [NIST 逐题及总体核对](evidence-final/scifact-product-v4-test-trec-crosscheck.json)、[300 题配对 bootstrap](evidence-final/scifact-product-v4-test-paired-bootstrap.json)、[测试前源码](evidence-final/scifact-test-source-before.json)与[测试后源码](evidence-final/scifact-test-source-after.json)。
- [BGE+Qwen 固定选择](evidence-final/scifact-product-v4-bge-qwen-selection.json)、[词法固定选择](evidence-final/scifact-product-v4-lexical-selection.json)、[v2 四组验证配对分析](evidence-final/scifact-product-v2-paired-bootstrap.json)、[v2/v3 输入等价及旧分数重放](evidence-final/scifact-v2-v3-input-equivalence-v2.json)、[缓存复制溯源](evidence-final/scifact-v2-v3-cache-migration.json)。同目录保留对应的 v4 验证报告、journal 和 v2 四组原始验证报告。

以上文件由任务输出目录逐字节复制，[复制完整性清单](evidence-final/copy-integrity.json)记录每项 SHA-256；它们只包含查询/文档标识、排名、分数、计时和运行元数据，不包含 BEIR 原始文档正文、模型权重、向量或认证信息。历史首次 trec_eval 缺少 `-c` 导致空排序查询被排除的失败记录保留在 [旧版校验记录](trec-crosscheck-incomplete-query-denominator.json)，不能作为最终指标。
