# 法律检索开发记录（未完成）

基线 `689f2a5bf4b853dfe93ac26802970f8996f993c8`，分支 `feat/versioned-legal-rag-20261105`。本目录不包含私有企业源代码和凭据。该候选是部分实现，不满足完整方案及生产验收。

## 已实现与运行

`legal_rag` 提供 PDF 页面合并、跨页条文定位、SQLite revision 隔离、条号与 jieba/FTS5 检索、RRF、仅本地权重的 BGE/Qwen CrossEncoder 包装、OpenViking find 请求包装、基于证据的模型流协议包装、少量 API、合成数据评测。无旧 torch/langchain 启动依赖。来源未审核时禁止激活；激活采用活动版本比较后事务更新。

在仓库根目录用独立 Python 环境安装 `requirements-v2.txt` 后：

```sh
python -m legal_rag.cli ingest --db /tmp/law.sqlite3 --pdf 'main-system/中华人民共和国民法典 - 中华人民共和国最高人民法院.pdf' --manifest docs/development/legal-rag-20261105/repository-source-manifest.json --revision repository-unverified
python -m legal_rag.cli evaluate --db /tmp/law.sqlite3 --revision repository-unverified --dataset docs/development/legal-rag-20261105/synthetic.json --variant rrf --output /tmp/evaluation.json
LEGAL_RAG_DB=/tmp/law.sqlite3 uvicorn legal_rag.api:app --host 127.0.0.1 --port 8000
```

本仓库 PDF 的来源未经核验，不能用上述 manifest 激活。服务身份必须由部署者在 `LEGAL_RAG_AUTH` 中配置 JSON 对象（token 映射到服务端用户 ID）；不要把密钥写入仓库。`/health/ready` 按活动索引、服务和模型可用性检查，缺项返回 503。不能仅改 manifest 中的验证状态冒充人工核验。

## 证据和边界

2026-11-05 UTC，隔离 `/tmp/legal-rag-candidate.sqlite3` 对仓库 PDF 提取到 1260 条；实际输入 SHA-256 为 `4cef5e4a1e5f82c5a52e949093ba0cfec4f7290e78d04d837c292e2a62c4065e`。来源仍待核验，不能将条数视为完整性或准确性的证明。首轮发现引用在换行开头导致条号重复，要求条号后有空白修复。早期测试后持续扩展，最终测试结果见下文。`evidence/eval-*.json` 为三个人造条号查询各执行 exact、BM25、RRF，Recall@20 均 1.0；这只能证明这些简单请求的命中，不能推断法律检索质量。流适配测试使用 mock HTTP 响应，未验证实际模型。

## 后续增量（2026-11-05）

上述早期记录中的以下缺口已部分补充：新增基于缩进的款项分段及章节边界识别（真实资料尚未逐条核对）；按生成模型 chat template 计数的完整条文预算接口（真实模型模板尚未验证）；管理员目录受限导入与持久 job 查询；SSE 接入可配置 OpenAI 兼容生成器，上游 delta 与后续引用结构校验事件分开，语义支持仍是未验证。缺少 tokenizer、向量索引验证或重排器时不会生成答案。服务端的每次检索使用独立 SQLite 连接，会话归属保存在数据库。

新增 `python -m legal_rag.viking_index send|reconcile|validate --help`：向 OV 发单元前先写 unknown 状态，失败不自动重试；回读 hash 一致才建立 URI 映射，全量单元通过范围内 find 可见性后才记录索引验证位。真实 OV 0.5.3 以单条合成资料完成回读和受限检索；法律语料未接入实测。新增 `python -m legal_rag.model_experiment --help`，BGE/Qwen 2×2 参数仅加载本地固定 revision；BGE 固定权重已完成三条合成问题的真实联调；Qwen 尚未完成模型实验。

最终测试与指标：45 项测试，真实 loopback uvicorn 子进程 6 项检查，详见 `evidence/tests.xml` 和 `evidence/subprocess-smoke.json`。首次 loopback smoke 被本机环境代理干扰，禁用环境代理后通过。300 个源文前 24 字生成的合成问题：exact 的 Recall@20 为 0；BM25 和 RRF 的 Recall@20 均为 1.0，nDCG@10 约 0.984。原文直接进入查询，存在严重数据泄漏，不是人工标注、真实问题或质量验收；每题查询、排序、指标、失败和耗时见 `evidence/eval-300-*.json`。模型生成流仅做 mock 测试；OV 另完成单条合成资料的真实后端联调。默认业务服务尚未连接完整依赖，readiness 返回 503；配置完整时会检查服务可达、模型身份与索引状态，但仍不能证明法律质量。

仍未完成：官方版本和 PDF 内容人工核验；条款、页眉、例外和转引关系全面核查；完整认证、生产审计和并发压力测试；真实法律语料的 OV/模型接入与性能；claim 语义验证、人工问题集及正式精度评估。这些新增能力均不等于真实业务验收。

增量：旧 API 提供默认关闭的 v2 适配且保留鉴权；新增可选持久会话正文及按服务端身份隔离的读取。完整操作、模型环境与隐私边界见 [RUNBOOK](RUNBOOK.md)。

补充合成集 `synthetic-diverse-300.json`：条号、摘录、词语、例外表达各 75 题，仍从答案原文派生，有直接文本泄漏，不能代替真实问法。按 exact、BM25、RRF 三种执行；RRF 总 Recall@20 约 0.997、nDCG@10 约 0.890，词语切片 nDCG@10 约 0.681，存在 1 个 Recall@20 未命中。每题预测、标注来源、失败和耗时、每类指标分别保存在 `evidence/eval-diverse-*.json`。

真实来源激活前现要求来源审核记录、原文 hash、FTS 条数一致与全量向量可见性；审核记录是人工声明而不是系统自动验证的官方证明。详见 RUNBOOK。

补充测试时发现测试用审核资料继承了 synthetic 类型，与新防绕过约束冲突；已将该测试输入标为 law 后重新运行，首次失败不计为最终通过。

补充防护：重排结果验证候选 ID 一致、数量与有限分数；不合格降级至 RRF，不能凭重排器注入新法条。

新增管理员导入测试实际处理仓库 PDF，确认 API 即使收到 approved 声明也强制 pending，job 可查询且重复提交幂等。来源本身依然未核验。

重排器按完整款项窗口打分、聚合法条；单款超过模型 4096 token 配置时拒绝静默截断并降级。模型和实际 tokenizer 行为仍待真实权重联调。

精确条号现支持全角阿拉伯数字、中文数字及“之一”的规范化匹配，保留原始显示条号。

最终代码重新导入隔离 PDF 得到 1260 条、2126 个候选款项和 1260 条 FTS 记录，见 `evidence/pdf-ingest.json`；这些计数并不证明款项或来源已被人工核验。

为避免旧应用启动时引入 torch/LangChain，新实现放在仓库根 `legal_rag/` 而非原方案的 `main-system/src/legal_rag/`；从仓库根启动，旧路径通过默认关闭的显式适配开关接入。明确条文原文与一般概念问题可 ready；个案缺日期需补事实，带日期时标记适用性待审核，不生成确定个案结论。jieba 搜索模式同时应用于索引与查询，parser/index 身份已更新；旧 `eval-300-*` 是之前版本的历史数据，不与新版直接比较。

一般概念生成需外部标定重排阈值；缺省拒绝生成。明确查询原文只返回命中的指定条号。阈值标定仍未执行。

新增合成空检索代理测试 `synthetic-empty.json`：两条明显域外合成问题及一条条号查询，三路均得到空检索代理 F1=1.0。样本极少，且该指标只统计是否返回候选，不等于回答层拒答 F1；不得用于正式质量判断。

新增同语料内部显式条号引用候选及文本偏移，全部标记未审核，不自动展开、不视为确定适用依据。新增 revision 保留上一活动版本，job 状态随验证和激活更新。

模型流现要求上游完成标记并限定总时长；流截断会返回错误而不标记完成。仍未进行真实服务断连/取消验证。

流式入口在响应头前校验会话权限与 revision，并固定该 revision；活动版本在响应期间切换时仍回读旧版本。已覆盖跨用户 403 与版本切换测试。

增加具名法律范围核验：请求刑法、劳动合同法等不同法律的同号条文时不返回民法典；覆盖书名号及若干常见表达，但名称识别并不穷尽所有法律别名，仍需人工范围核验。

引用详情支持显式读取已审核且保留的旧 revision；未激活的暂存版本不可通过该接口读取。

2×2 模型实验脚本现执行 exact/BM25/dense 的 RRF 候选融合，再按完整款项重排；替身模型检查了执行流程和候选来源，不构成真实模型结果。

活动版本切换前重新核对全文索引每条分词内容、ID 与原文 hash，发现内容错位时拒绝激活；该过程仍不能验证原文的官方真实性。

OV 适配器还通过真实 loopback HTTP 假服务覆盖 multipart 上传、导入、回读和 find 契约；它不是 OpenViking 引擎，不能证明真实引擎兼容。

对当前隔离语料独立重算了全部 1260 个原文 SHA-256 与 FTS 分词内容，0 条内部不一致，见 `evidence/catalog-consistency.json`。这只验证本地索引与提取文本的一致性，不证明原文准确或来源官方。

候选保留各路 rank 和可用原始分数，FTS5 BM25 明确标注越小越优、OV 分数标记为后端原始值；RRF 按名次融合，不把分数当概率或直接跨源比较。

JUnit 文件中的本机 hostname 已脱敏；测试 ID、结果、时间和错误统计保留。证据不含认证令牌或真实会话。

明确查询某条原文时，如果指定条号任何一条不存在，返回证据不足，不用近似条文替代或把部分命中冒充完整结果。

严格 TLS 获取 [天津通信管理局民法典全文](https://tjca.miit.gov.cn/zwgk/zcwj/flfg/art/2020/art_20cf1a2e1b854924b5caa744c8045d1f.html) 和 [市场监管总局民法典全文](https://www.samr.gov.cn/zw/zfxxgk/fdzdgknr/bgt/art/2023/art_60fb3863b7364b759123a42d1df568ca.html)，各 HTML 软件按段落解析均有 1260 条。修复前两者与仓库 PDF 提取文本各 1259/1260 归一化一致，差异为第1258条末尾误附“附则”；修复后各为 1260/1260 一致。去空白及 NFKC 会忽略格式差异；两页是政府转载而非原始颁布载体，尚未人工逐页、效力、再分发审核，语料仍 pending。证据见 `evidence/official-alternates-fetch.json`、`official-crosscheck-before-fix.json`、`official-crosscheck.json`，复现脚本见 `tests/v2/check_official_sources.py`。人大 PDF TLS 握手失败、最高法院页面 TLS EOF、福州 PDF 超时均记录在同一 fetch 证据中；未绕过 TLS。

OpenViking 0.5.3 真机 HTTP 集成：实际 `to` 是目录，L2 路径为 `root/<unit_id>.md`，首轮误解导致回读不符，已修正适配器。首次导入因本机系统代理导致向量服务连接失败并超时，保留 delivery_unknown；只读确认其目录及 L2 hash、原任务仍 running，未重放原请求。第二个隔离 workspace 配置 loopback NO_PROXY、使用全新合成 revision，真实 BGE-M3 1024 维 embedding 下完成导入、原文 hash 回读、scope/level=2 find 和向量可见性检查，见 `evidence/ov-real-smoke.json`；仅一条合成资料，不是法律质量验收。

`evidence/bge-real-smoke.json` 是固定 BGE-M3 + BGE reranker 在 MPS float16 上的 3 条合成条号流程验证，先后加载模型、标注实际设备及 dtype；不能作为业务准确率。Qwen 权重、实测和 chat template 对照仍待验证。

真实 v2 HTTP 服务另使用合成语料验证了认证缺失 401、exact/BM25/真实 OV dense 同时命中、真实 BGE reranker 和 source lookup；单次响应约 331ms，未加载生成模型或 tokenizer，readiness 503。见 `evidence/v2-real-http-smoke.json`；该数据不是性能统计或法律问答验收。
