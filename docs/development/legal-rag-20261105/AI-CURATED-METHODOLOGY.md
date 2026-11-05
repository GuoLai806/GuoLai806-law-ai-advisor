# AI 整理的民法典检索题：方法与边界

这组题用于在固定语料上发现检索问题，**不是人工标注的专业基准，也不是法律意见质量证明**。问题、必要依据和简短理由均由 AI 根据条文编写；所有条目标注为 `author: AI`、`review_state: needs_human_review`。题目不是从现有 300 条“原文前缀”合成集改写，但作者已读到来源和答案，因此属于**来源知情的独立编题**，不是独立于语料的真实用户样本或双盲题。不得将其称为真实用户问题、人工审核金标准或专业法律评测集。

## 语料和逐题依据

候选文件 [ai-curated-eval.json](ai-curated-eval.json) 使用 SQLite `repository-unverified`、解析器 `article-v4-section-crosscheck` 中的 2020 年民法典 PDF 提取结果。PDF SHA-256：`4cef5e4a1e5f82c5a52e949093ba0cfec4f7290e78d04d837c292e2a62c4065e`。本地版本仍为 **pending**；这里不声称完成来源授权、现行效力或人工逐页审核。第 1260 条有施行日期，但这一份 2020 年文本不能单独证明任意后续日期法律未修订。需另外核对适用时间、特别法及司法解释。

已有的 [逐条机器交叉检查](evidence/official-crosscheck.json) 对 PDF 提取条文与 [天津通信管理局转载全文](https://tjca.miit.gov.cn/zwgk/zcwj/flfg/art/2020/art_20cf1a2e1b854924b5caa744c8045d1f.html)、[市场监管总局转载全文](https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_60fb3863b7364b759123a42d1df568ca.html) 进行了 NFKC 加去空白的程序化比较：两份转载各 1260/1260 匹配。它不是颁布机关原始载体、逐页人工核验或现行效力证明。原始颁布载体获取失败等情况保留在已有证据中，未绕过 TLS。

每条有依据的题都保存条文 label、unit ID、原文 SHA-256、PDF 页码/字符位置和解释该条为何相关的简短理由；对事实不足题，这些是可支持一般规则或追问的背景依据，**不代表具体个案结论已成立**。库外题没有伪造的民法典来源。对全部 127 个引用实例重新读取本地库并重算文本 hash 与 locator，见 [校验结果](evidence/ai-curated-reviewed-validation.json)。这只是结构和文本一致性检查，不是条文适用性的独立人工复核。已逐项对照条文进行 AI 自查，后续仍需法律人员逐题判断题意、必要性、例外、事实缺口和可能的其他充分依据。

## 内容与拆分

共 120 题：105 条拟按给定语料解释的一般规则问题（其中 2 条必要依据存在争议，暂不参加正向评分）；15 条要求拒绝作确定结论或追问，其中 11 条是刑事、劳动、税收和行政程序等库外问题，4 条是关键事实不足。事实不足并不要求一律拒绝说明一般规则；可以说明受支持的原则并询问缺失事实，不能虚构裁判、计算结果或把不确定性变成必然结论。

| 划分 | 总题 | 可检索规则题 | 需追问或避免确定结论 |
|---|---:|---:|---:|
| train | 74 | 69（另 1 条待裁决） | 4 |
| dev | 22 | 15（另 1 条待裁决） | 6 |
| test | 24 | 19 | 5 |

分组按具体主题固定，再验证同一引用条文和同一 topic 不跨分组；因此同一条文的相似问法不会分别进入开发和测试。不同主题仍可能共享法律概念，不能宣称统计独立。覆盖总则、物权、合同、人格权、婚姻家庭、继承、侵权、版本及库外/事实不足；题型标签中的多条、否定或例外问题都仍需人工确认。类型数量是编题时的简单标签，不代表实际用户频率。将问题与其标注条文的文本分别做 NFKC 并移除非字母数字后，未发现连续 14 字符及以上完全相同的片段；这是机械抄录检查，不证明提问自然、没有较短共享术语或没有语义泄漏。

`expected_labels.required` 表示本次 AI 标注的最小必需条文，`alternatives` 是保留给经过审核的替代依据组；目前没有足够可靠的可替代组，因此为空，**不代表其他法条不相关或不可能构成充分依据**。当前只有二元相关性，不存在人工 0–3 分分级。评测结果必须保留每题原始排名、未命中列表、分子与分母；不得凭这一数据宣称某模型具有法律准确率、司法胜率或可以给出个案结论。

## 冻结、纠错和评测

根据独立 AI 审阅修订的开发候选 SHA-256：`381bc2db3d2a4c093d781b41fd2d5cd3172bd4b99d00a9d0eb9cca05ea645ae6`；开发投影 SHA-256：`2b880f6f0999158a2f975d83a3bae7302e0787df1b6369c2809344621bacd2bd`。两者的前版按字节保存在 evidence 目录，修改原因和新旧依据见 [独立审阅修订记录](evidence/ai-curated-review-corrections.json)。独立 AI 审阅者已读过全部题目，包括原 `test` 组；**全体均为开发数据，任何组都不是独立盲测或最终测试**。原 `test` 组尚未用于模型运行，但已被作者和审阅者看到，不可据此声称未接触的测试集。

第一次编题版本曾在保留副本之前被覆盖，原始字节未能恢复，其记录 hash 为 `9adb45a755708ae55a13eae9647ac9ab11a8dee1618ca6c0eaf5e23cf5df44f1`。幸存的编题种子重建了初版内容，但重建版的元数据时间与 hash 不同，不能冒充原件；初版 dev 投影已重建并与当时记录的 `3d9cf4429e0ce04fad15cc0c15552ca04fabc811f896d6cd125fbd174d23a83b` 字节 hash 一致。四处修改及原因记录在 [纠错记录](evidence/ai-curated-corrections.json)，包含“撤回/撤销”区别、行为能力和适用场景限定。随后对三条事实不足问题进行语义调整，避免把可以直接回答的一般性问法误标为必须追问；保留上一候选和 [第二轮纠错记录](evidence/ai-curated-abstention-corrections.json)。现有记录中未对初版进行模型评测；原 test 组已由作者和独立 AI 审阅，但尚未用于模型运行。

[适配与评分脚本](../../../tests/ai_curated_metrics.py)只投影标注尚未发现必要依据争议的正向题供现有检索实验脚本使用，按文件 hash 核对结果对应的投影；不会把 15 条应追问/库外题当作空结果检索准确率，也不会凭“无候选”冒充回答层拒答。单独的回答层实验必须保存真实生成结果并由合适的人工标注判定追问、拒答与部分解释；在那之前不报告 Abstention F1。原 test 分组的投影需要 `--include-reviewed-test` 明确选择，但它仍只是已审阅的开发数据，不是最终盲测。

从仓库根目录运行结构校验或重新投影开发集（实际数据库路径由执行者配置）：

```sh
python tests/ai_curated_metrics.py --dataset docs/development/legal-rag-20261105/ai-curated-eval.json --expected-sha256 381bc2db3d2a4c093d781b41fd2d5cd3172bd4b99d00a9d0eb9cca05ea645ae6 validate --catalog "$CATALOG_PATH" --output /tmp/ai-curated-validation.json
python tests/ai_curated_metrics.py --dataset docs/development/legal-rag-20261105/ai-curated-eval.json --expected-sha256 381bc2db3d2a4c093d781b41fd2d5cd3172bd4b99d00a9d0eb9cca05ea645ae6 project --split dev --output /tmp/ai-curated-dev.json
```

评分脚本的 hash、漏项、重复题和 test 开关自检采用构造的假排名，见 [合成自检](evidence/ai-curated-reviewed-metric-selfcheck.json)，不是真实模型成绩。真实比较需在相同语料和检索配置下保留模型及权重 revision、候选、每题排序、超时与失败、设备、耗时和完整原始报告；小样本应报告计数和不确定性，不使用固定 15 或 19 题的单次百分比作为生产能力证明。


独立 AI 审阅发现，原分组中库外领域高度绑定：原 test 库外题均是刑事，dev 库外题是劳动与税收，train 库外题是行政程序。领域词本身可能提示标签，不能用此分组评判域外拒答的泛化。问题 101、102 的最小充分条文尚有争议，保留原观点及候选依据但标为 `metric_eligible:false`；在法律专业人员裁决前不计入正向检索指标，也不声称已经验证了其他题的充分性。作者没有新增所谓盲测，也没有以模型结果调整这些问题。
