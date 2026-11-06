# 行业参照：检索接近哪些公开结果，哪些还不能比较

**本项目在SciFact上的检索排序处于本次外部参照的较高区间，但不能据此称完整RAG达到业界前沿。** 正式融合/重排nDCG@10为0.794201，接近本次查到的NV-Embed-v2 0.80111和stella1.5B 0.80090；本项目Recall@20为0.935889，低于这两份公开报告的0.95333和0.96167。完整问答只有269/300题完成、标签一致率70.67%、自定义引用文档标签F1为0.535774，这些能力没有同口径外部完整系统对照。

用户说明见 [用户版](PLAN-FOR-YOU.md)，全部正式质量与失败见 [结果报告](RESULTS.md)，统计边界见 [评测口径](METHODOLOGY.md)。本文中的“公开结果”指固定来源的外部报告，不是本项目在同一机器复跑出的成绩。

## 调查范围和固定来源

来源为 [embeddings-benchmark/results固定提交](https://github.com/embeddings-benchmark/results/tree/922b3df446f88756d954e4099b7211bcafaf99c9)，commit为 `922b3df446f88756d954e4099b7211bcafaf99c9`。调查明确列出38个通用英文/文本模型配置，保留查询项、匹配项和缺失状态；其中14份原版SciFact报告的task与dataset revision一致，可以列作检索参照。

这一组采用task `SciFact`、dataset revision `0228b52cf27578f30900b9e5271d331663a030d7`。翻译集、不同revision、SciFact Nano与LMEB任务没有拼进下表。38个配置不是全球全部模型，因此不能说这是最新全行业排名或任何系统的全局第一。

每个模型行都链接到固定commit下的原始JSON，调查还记录模型revision、MTEB版本、数据revision、source路径与blob/hash。`external`表示来源仓库中的external条目，不是本项目独立复现。完整调查见 [38配置来源记录](evidence/industry-survey.json)。

## 14份同task/revision公开检索结果

按外部报告的nDCG@10排序；小数保留其报告精度，不推算未报告的Recall：

| 外部模型与原始JSON | nDCG@10 | Recall@20 | 来源形式 |
| --- | ---: | ---: | --- |
| [NV-Embed-v2](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/nvidia__NV-Embed-v2/7604d305b621f14095a1aa23d351674c2859553a/SciFact.json) | 0.80111 | 0.95333 | 模型revision条目 |
| [stella_en_1.5B_v5](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/NovaSearch__stella_en_1.5B_v5/external/SciFact.json) | 0.80090 | 0.96167 | external条目 |
| [GritLM-7B](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/GritLM__GritLM-7B/13f00a0e36500c80ce12870ea513846a066004af/SciFact.json) | 0.79129 | 0.94833 | 模型revision条目 |
| [gte-Qwen2-7B-instruct](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/Alibaba-NLP__gte-Qwen2-7B-instruct/external/SciFact.json) | 0.79060 | 未报告 | external条目 |
| [Qwen3-Embedding-8B](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/Qwen__Qwen3-Embedding-8B/4e423935c619ae4df87b646a3ce949610c66241c/SciFact.json) | 0.78457 | 0.95000 | 模型revision条目 |
| [Qwen3-Embedding-4B](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/Qwen__Qwen3-Embedding-4B/636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff/SciFact.json) | 0.78333 | 0.95000 | 模型revision条目 |
| [stella_en_400M_v5](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/NovaSearch__stella_en_400M_v5/external/SciFact.json) | 0.78233 | 0.94833 | external条目 |
| [e5-mistral-7b-instruct](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/intfloat__e5-mistral-7b-instruct/07163b72af1488142a360786df853f237b1a3ca1/SciFact.json) | 0.76417 | 0.91489 | 模型revision条目 |
| [jina-embeddings-v4](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/jinaai__jina-embeddings-v4/26239889730c735ed7e9a4db9180c8935faf4ba0/SciFact.json) | 0.76101 | 0.91833 | 模型revision条目 |
| [mxbai-embed-large-v1](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/mixedbread-ai__mxbai-embed-large-v1/990580e27d329c7408b3741ecff85876e128e203/SciFact.json) | 0.73917 | 0.93056 | 模型revision条目 |
| [jina-embeddings-v3](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/jinaai__jina-embeddings-v3/215a6e121fa0183376388ac6b1ae230326bfeaed/SciFact.json) | 0.72533 | 0.90033 | 模型revision条目 |
| [snowflake-arctic-embed-l-v2.0](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/Snowflake__snowflake-arctic-embed-l-v2.0/edc2df7b6c25794b340229ca082e7c78782e6374/SciFact.json) | 0.70896 | 0.86589 | 模型revision条目 |
| [Qwen3-Embedding-0.6B](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/Qwen__Qwen3-Embedding-0.6B/b22da495047858cce924d27d76261e96be6febc0/SciFact.json) | 0.69718 | 0.88667 | 模型revision条目 |
| [BGE-M3](https://github.com/embeddings-benchmark/results/blob/922b3df446f88756d954e4099b7211bcafaf99c9/results/BAAI__bge-m3/5617a9f61b028005a4858fdac845db406aefb181/SciFact.json) | 0.64366 | 0.82356 | 模型revision条目 |

本项目真实OpenViking正式检索链另列如下，避免将系统配置差异藏在同一榜单里：

| 本项目阶段 | nDCG@10 | Recall@20 | 配置 |
| --- | ---: | ---: | --- |
| OpenViking dense单路 | 0.644566 | 0.823556 | BGE-M3，真实OpenViking |
| 完整检索排序 | 0.794201 | 0.935889 | 关键词 + OpenViking/BGE-M3 + RRF + Qwen3-Reranker-4B |

本项目dense单路与公开BGE-M3结果数值相近，完整检索链则增加了关键词和重排计算。0.794201与NV-Embed-v2相差约−0.006909，与stella1.5B相差约−0.006699；但Recall分别相差约−0.017441、−0.025781。不能只选nDCG比较而隐藏Recall差距。

## 有较强内部证据的提升是什么

本轮300题的真实排名trace完整，可在相同输入与qrels下比较排名阶段。相较BM25，最终nDCG从0.668898提高到0.794201，绝对差0.125303、相对提高18.73%；相较OpenViking/BGE-M3单路，绝对差0.149634、相对提高23.21%；相较RRF重排前，绝对差0.089927、相对提高12.77%。

这些差值按同query配对重采样得到的95%区间分别为[0.092150,0.159107]、[0.116788,0.183612]、[0.061724,0.120235]；同一次运行里可以看到排序贡献。全部300题保留，未挑提高的题目：相较RRF重排前仍有26题的nDCG下降。NIST trec_eval v9.0.8逐题与总体复核舍入一致。详细Recall/MRR与原始记录见 [阶段配对证据](evidence/openviking-formal-test-v1.retrieval-paired.json)。

它不是三个完整答案系统各自运行一次的A/B，也不是对外部模型的显著性检验。不能把18.73%的排名提升说成答案正确率提升18.73%，不能把加入更多重排算力的方案与单embedding模型包装成公平的总成本/速度胜出。

## 为什么现在还不能写完整系统SOTA

外部表主要提供检索分数。本项目有不同的系统组成、模型计算量、硬件、软件版本、提示方式和流程，不在同一环境独立复跑。外部没有这份材料内的同口径最终标签、语义引用、拒答、保存和用户等待数据；不能因为一个检索分数接近就推出完整答案质量接近。

本项目nDCG的query重采样95%区间为[0.757479,0.829855]，反映这批题的抽样不确定性；同文献问题可能相关。这个区间包含某个外部均值，不证明等效、显著超过或无差异。没有外部逐题同口径运行结果，不能作外部配对检验。

本轮31/300题未完成，其中17题只展示证据、14题记录生成阶段超时/协议/连接错误；112题证据不足类只判对58题。即使188道正类里172道找齐了全部标注证据，仍有19道标签错误、9道流程失败。引用检查只是编号与来源一致，语义支持尚未逐句人工裁决；自定义文档与标签F1不能与官方SciFact句子rationale榜单混用。

正式期间主机发生14,756秒的日志Sleep交集；UTC总跨度25,073秒，逐题macOS单调时钟之和10,378.623392秒。现有延迟不能证明持续唤醒机器的生产用户等待、吞吐或高并发。睡眠与连接错误的具体因果没有被证明，也没有用这个主机条件删掉错误题。详见 [主机计时审计](evidence/host-timing-audit-v1.json)。

最后，科学摘要论断核验只是一个领域，且使用SciFact公开dev标注，不能称隐藏榜单成绩。中文、跨领域、长文、复杂PDF/OCR、多轮问答和生产并发尚需独立评测。

## 在文档和简历里可以怎样准确表述

可以写：“在SciFact 5,183篇文献、300题真实OpenViking流程评测中，检索nDCG@10为0.7942、Recall@20为0.9359；相较同query的BM25排名，nDCG相对提升18.73%，保留失败、用量及版本审计。”若要介绍外部对照，可以写“检索nDCG接近本次调查的部分公开高分embedding结果”，紧接着注明混合检索加重排与单模型方案不同。

目前不能写“业界第一”“完整RAG达到SOTA”“OpenViking使答案准确率提升18.73%”“引用零幻觉”“全部300题流程成功”或将受主机睡眠影响的分位写成已验收生产延迟。外部表是有来源的参照，能支撑范围有限的检索讨论，不能替代完整系统的质量证明。

要证明下一版达到完整系统前沿，需要事先冻结同一任务、同一输入、同一gold/评分、同一失败和用量口径，重跑强基线，另做语义引用与拒答评估，并在持续唤醒、明确负载条件下比较用户等待和资源。扩展到通用RAG需增加不同领域与长文数据；新协议保留本轮全部成绩，不能用正式错题反复调到满意再冒充独立测试。
