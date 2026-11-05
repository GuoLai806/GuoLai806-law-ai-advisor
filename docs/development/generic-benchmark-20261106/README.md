# 通用检索公开基准：冻结方案与已验证边界

最终状态（2026-11-06）：v4选定配置和产品词法基线均已完成官方SciFact 300/300 test、零执行故障，并通过NIST逐题/总体核对、journal完整性和源码前后hash检查。正式结果见 [RESULTS.md](RESULTS.md)，以下v2/v3说明保留候选选择和修复过程，不是当前未完成状态。

这套实验以通用检索为主，使用官方 BEIR 的 SciFact 和可选 NFCorpus；不是民法题集，也不评估生成答案、法律咨询、问答 EM/F1 或完整产品使用体验。SciFact 是英语科学文献检索，NFCorpus 是英语营养与医学检索，两个领域不能代表其他语言、一般网页或企业私有材料。

## 来源、许可和固定查询

- [BEIR 固定版本 README](https://github.com/beir-cellar/beir/blob/ef83d29307061c65d04b035b4f4e7c18bd8374af/README.md)提供下载地址和 MD5。SciFact 官方包 MD5 `5f7d1de60b170fc8027bb7898e2efca1`，下载后的 SHA-256 `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`；全库 5,183 条，官方 train 809 个查询、test 300 个查询。BEIR 数据卡标明 [CC-BY-SA-4.0](https://huggingface.co/datasets/BeIR/scifact)，仍应保留署名和适用许可信息。
- NFCorpus 官方包 MD5 `a89dba18a62ef92f7d323ec890a0d38d`，SHA-256 `efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b`；BEIR 子集 3,633 条，dev 324、test 323 个查询。BEIR [数据卡](https://huggingface.co/datasets/BeIR/nfcorpus)写 CC-BY-SA-4.0；原始 [Heidelberg 数据发布方](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)同时说明学术使用，以及其他 NutritionFacts 数据用途须查看原网站条款并联系作者。此处保留两种来源陈述，不推断商业使用或再分发授权。
- 下载器校验压缩包 MD5、记录 SHA-256，拒绝路径穿越、符号链接和过大的归档；检查文档/查询 ID 唯一、qrels 引用完整及固定数量。原始语料、压缩包、模型权重和向量缓存放在仓库外，不进入提交。
- SciFact 从官方 train 的 809 个有标签查询中按 SHA-256(query ID) 排序选 32；NFCorpus 从官方 dev 中按同法选 32。查询 ID、语料及 qrels 哈希保存在 `benchmarks/manifests/`，在模型运行前已冻结。小验证集用于选择配置，不是正式测试集。正式 SciFact test 必须完整运行 300 题；正式结果已完成，见RESULTS.md；本节记录此前冻结方法。

## 产品共享路径与范围

候选比较时的验证配置是 [product-validation-frozen-v2.json](../../../benchmarks/manifests/product-validation-frozen-v2.json)，SHA-256 `12ebd86d58ea9791f9fe150d38b5860fe405a546610accf24b11f46d122605e8`。它锁定验证清单、通用产品源文件及脚本哈希、模型 revision、平台、库与 jieba 词典版本、MPS float16、batch 2、各路 top 40、RRF k=60、候选上限 60。冻结时刻和第一条产品词法验证的时间关系见 [本机记录](product-freeze-attestation.json)；这是本机文件和日志证据，不是可信时间戳服务。

实验把完整语料交给产品 `evidence_rag.catalog.Catalog.stage_documents`，用同一个 `lexical_scored`、`rank_candidates`、RRF 和 `CrossEncoderReranker`，按 catalog 中文档 ID 映射到官方 qrels。通用标题精确匹配来自查询和文档标题，未使用 qrels 或黄金 ID。嵌入文本通过与产品 OV L2 写入相同的 `document_embedding_text` 构造：标题、换行、正文；不附加 doc ID、答案或相关性。空标题产品回退到 ID 的词法污染已修复为正文，并用测试检查。修复前的冻结配置和结果保留为历史，当前结果使用新配置；SciFact 无空标题，前后全部排序及来源候选完全一致，见 [等价检查](scifact-product-v2-equivalence.json)。

密集检索在完整语料上计算本机归一化向量的精确余弦，是**参考后端**；本实验不等于 OpenViking 已索引完整 BEIR。候选顺序与分数来自产品共享排序核心的**证据预算前**阶段；产品后续可能按段落选择、预算限制、证据数量过滤，不能把该排序当成最终证据或回答准确率。产品的真实 OpenViking/HTTP 小样本证明另行记录，不能把它扩大成 5,183 条 OV 检索实测。

基准缓存位于仓库外，键包括语料和文档顺序哈希、输入格式及序列化代码、模型与 revision、设备和 dtype、归一化、维数、查询及指令哈希与 batch；`.npy` 用 `allow_pickle=False` 加载并核对内容哈希、维度、有限值和单位范数。文件锁避免同键并发写入。先加载/释放嵌入模型，再加载/释放重排模型；超长输入或设备/精度不符不静默截断或降级，逐查询失败计零并在报告中标记不具可比性。原始每题排名、各路分数、失败、计时、缓存和预热均保留。查询向量按 batch 2 计时，不能当作独立单题的端到端延迟；首次模型加载和建索引与暖缓存分开。

## 参考与产品词法验证

下表只覆盖各自固定的 32 条验证查询；所有值均为观察到的检索排序统计，不是最终答案质量。统计中没有因空排序丢掉查询。

| 语料及路径 | nDCG@10 | Recall@20 | MRR@10 | 请求失败 |
|---|---:|---:|---:|---:|
| SciFact 独立 unicode61 FTS5 参考（非产品路径） | 0.74609 | 0.83594 | 0.74089 | 0/32 |
| SciFact 产品词法 + 共享候选排序（新冻结版本） | 0.74564 | 0.83594 | 0.74045 | 0/32 |
| NFCorpus 独立 unicode61 FTS5 参考（非产品路径） | 0.19207 | 0.12447 | 0.36384 | 0/32 |
| NFCorpus 产品词法 + 共享候选排序（旧冻结版本） | 0.19221 | 0.12447 | 0.36488 | 0/32 |

新配置下的 NFCorpus 产品词法尚未重跑，上行不得作为新配置成绩。参考路径与产品路径分开，不把其结果称为产品指标。指标按 BEIR 的结果端同 ID 过滤、TREC `ndcg_cut` 的**线性相关性等级**、正相关 Recall 和前 10 首次命中 MRR 计算。用本地隔离编译的 [NIST trec_eval v9.0.8](https://github.com/usnistgov/trec_eval/tree/v9.0.8) `-c` 对两份独立参考结果和 SciFact 产品排序交叉核对，误差在其四位小数舍入范围内；后者将产品顺序编码为 rank 分数，避免额外改变平分排序。第一次 NFCorpus 交叉核对漏加 `-c`，4 条无排序查询被排除造成分母偏小；失败记录和纠正均保留在 `trec-crosscheck-incomplete-query-denominator.json`、`trec-crosscheck.json`。空结果按零计入，执行异常同样计零并使运行不具可比性。

## 运行与验证

设置 `BENCH_DATA_DIR`、`BENCH_CACHE_DIR` 和 `HF_HOME` 为仓库外的私有本机目录，在安装好锁定模型依赖的隔离环境从仓库根目录运行。模型使用 `local_files_only`，不会隐式下载。以下是 SciFact 验证集的一组；分别替换 `bge|qwen` 得到四组，**顺序运行并使用不同输出文件**：

```sh
python -m benchmarks.run_product --data-dir "$BENCH_DATA_DIR" \
  --manifest benchmarks/manifests/scifact-validation.json \
  --freeze benchmarks/manifests/product-validation-frozen-v2.json \
  --split validation --mode product-hybrid --embedding bge --reranker bge \
  --device mps --dtype float16 --batch-size 2 --cache-dir "$BENCH_CACHE_DIR" \
  --output /tmp/scifact-bge-bge-validation.json
```

完成验证并选定具体配置后，用 `python -m benchmarks.select --manifest ... --validation-report ... --output ...` 固定报告哈希、配置和源代码哈希；正式 test 命令必须传相同 `--selection` 与 `--validation-report`，并由运行器检查一致。配置或源码变更会拒绝沿用旧选择；如确需变更，保存旧版本与失败记录，重新冻结、只在验证集选择，不根据测试结果反复挑选。运行器不能阻止一个有本地写权限的人手工伪造选择文件；这是工程流程约束而非防篡改系统。

合成契约测试和它们的限制见 [JUnit](tests.xml) 与 [日志](tests.log)。当前未执行 SQuAD 2.0 生成问答/拒答评估，不提供 EM、F1 或无答案成绩。任何模型运行后的结果要看其记录的候选覆盖、失败数、模型与源哈希；没有完整结果不能把计划中的 2×2 表述为已经完成。

开发时有一次合成测试失败：测试用开发虚拟环境未安装 `torch`，但虚拟模型测试试图读取真实 Torch 包版本，导致 `PackageNotFoundError`；改为为替身显式注入合成版本后，15 项测试通过。该失败发生在合成测试，不是真实模型运行；当时工具输出中有错误，未单独保存原始日志，不应冒充已归档原始记录。真实模型运行、预热和失败以各次独立报告为准。
