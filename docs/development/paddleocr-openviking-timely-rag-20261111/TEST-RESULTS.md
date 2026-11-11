# 测试结果与证据口径

更新于 2026-11-11。**当前结论**：民法典解析、结构化索引、时间检索、真实 API 与安装包已跑通；通用法律质量、重排和生成回答还没有达到可写“业界前沿”的证据。

| 项目 | 最终结果 | 原始证据 | 口径 |
|---|---:|---|---|
| 后端 v2 | 503 passed，5 warnings | [pytest-v2-final.log](evidence/pytest-v2-final.log) | 当前源码全量回归；warning 未省略 |
| 前端 | 11 passed | [node-v2-final.log](evidence/node-v2-final.log) | 日期、版本、异步状态、SSE 切帧和计时 |
| Python 编译 | 通过 | 本轮 py_compile | `evidence_rag/*.py`、`benchmarks/*.py` |
| diff / whitespace | 通过 | 本轮 git diff --check | 不代替语义评审 |
| wheel | 126,583 bytes，SHA-256 `589ee363e708029395d3a534cdc04ff58ced6aca26dcdda68faa845865ecec08` | [wheel-and-install-final.json](evidence/wheel-and-install-final.json) | 构建成功；模型和 DB 不入包 |
| wheel 安装 | 通过 | [wheel-install-final.log](evidence/wheel-install-final.log)、[cli-help-final.log](evidence/cli-help-final.log) | 从 site-packages 导入 evidence_rag/benchmarks/legal_rag，CLI help 退出 0 |
| 官方来源 | 唯一 ID，标题/机关/公布日/施行日已核验；目录树 1260 条 | [npc-official-metadata.json](evidence/npc-official-metadata.json) | 官方 PDF 下载返回网页壳；正文来源单独标识 |
| PaddleOCR | 64/64 页，110,753 字符，0 空页；1260/1260 边界；1259/1260 字面一致；1 条复核 | [ocr-summary-final.json](evidence/ocr-summary-final.json) | 最低行置信度 0.875195；0.992975 相似度是 difflib 代理，不是人工 CER |
| OpenViking journal/send/reconcile | 1260/1260 content verified，0 失败 | [openviking-index-summary.json](evidence/openviking-index-summary.json) | OpenViking 0.5.3；耗时 201.49 秒 |
| OpenViking 可见性 | 1260/1260 visible，排名均 1 | [openviking-validate-summary.json](evidence/openviking-validate-summary.json) | 用自身正文找回自身 URI 的完整性验证，不是检索质量 benchmark |
| 时间正式集 | 30 题，时间过滤后 nDCG@10 .987698、Recall@20 1.0、stale leakage 0 | [temporal-formal-results.json](evidence/temporal-formal-results.json) | 作者构造、非独立人工法律集；各题行和延迟保留 |
| 真实 API | 早于施行日返回 0 条；有效期内返回 8 条有效证据；explore 原文 hash 回读一致 | [api-e2e-summary.json](evidence/api-e2e-summary.json) | 只运行到检索/证据；模型未配置时 chat 只给 fallback |
| 证据文件 | 逐文件 SHA-256 | [SHA256SUMS.txt](evidence/SHA256SUMS.txt) | 公开的脱敏聚合结果 |

## 正式 30 题对照

同一 1260 条 revision、同一查询和固定参数：

| variant | nDCG@10 | Recall@20 | MRR@10 | 有效版本 Hit@10 | stale leakage | 平均阶段耗时 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | .365981 | .5 | .333333 | .933333 | 1.0 | 8.34 ms |
| OpenViking+BGE-M3 | .446426 | .5 | .427778 | 1.0 | 1.0 | 18.91 ms |
| RRF | .487698 | .5 | .483333 | 1.0 | 1.0 | 19.16 ms |
| RRF+Timely | **.987698** | **1.0** | **.983333** | **1.0** | **0.0** | 19.16 ms |

“stale leakage 1.0”表示没有任何时间过滤的三种基线都在无有效版本应返回空的题中返回了民法典条文；Timely 路径把 10 个施行前及 5 个历史问题全部拦住。正式集只包含时间边界、一般当前和已知历史问题；作者构造使“满分”不能当成公开泛化能力。

开发集 6 题同样保留：[temporal-development-results.json](evidence/temporal-development-results.json)。没有用正式标签调 alpha、候选上限或阈值。

## 失败、未执行与限制

- Qwen3-Reranker-4B、BGE-reranker-v2-m3 本地只有部分元数据，权重缺失；独立加载试验真实报错，故没有 cross-encoder 列。历史 SciFact 结果明确排除。
- 本机没有生成模型配置；`/api/v2/chat` 返回 evidence-only fallback，`usage=null`，不能作为问答质量或 token 账单。
- `/health/ready` 为 503 是预期：缺 reranker、generation budget、answer model；检索 API 仍可独立工作，不能把 readiness 假报成功。
- 唯一 OCR 差异条为“第一千零八十五条”，错误为“判决”→“决”；正文仍保留复核状态，不宣称零错。
- 官方元数据和正文不是同一次全链路下载；PDF 端点返回 HTML 壳，代码按文件签名拒绝。
- 只有 2021-01-01 起的一个真实 revision；历史法律、未来修订、司法解释、特别法、追溯/过渡条款还没有完整数据。
- 503 项测试存在历史测试复用，不等于 503 个互不重复的用户验收；API/UI 的真实模型生成和并发未验。
- 任何“业界前沿”“SOTA”“通用法律准确率 98.77%”都超出了本轮证据。

## 复现入口

- 正式 benchmark：`python -m benchmarks.run_legal_temporal`。
- 开发题和正式题的本机冻结文件为 `work/dev/civil-time-rag-20261111/benchmark/development-6.json` 与 `formal-30.json`；报告内保存数据集 SHA-256。公开证据包含逐题脱敏行，本机临时题集不提交。
- OpenViking 全量索引：`python -m benchmarks.index_legal_openviking index|validate`，配置必须匹配 revision scope。
- 数据、模型权重、OCR 页图和 SQLite 都保留本机，公开库只有代码、脱敏聚合报告和哈希。
