# 通用 Evidence RAG 独立只读评审

评审范围：通用 profile 的 API、生成、预算、目录、检索与产品基准口径；法律 profile 仅作可选路径边界检查。使用临时目录、假 provider、假重排器和本地 SQLite；未启动 MPS/GPU、GUI、付费模型或封存测试集，也未修改源码、基准文件或检索矩阵。报告描述测试时实际观察；矩阵运行期间相关源码的后续修复和重测由主任务负责。

## 需求符合性与质量结论

- 产品默认入口是通用 Evidence RAG，法律是可选兼容 profile；认证、会话归属、引用结构校验和失败降级有实现。`claim_support=not_verified` 明确指出引用语法有效不代表主张有语义支持，因此不能据此声称回答或法律结论正确。
- 基准 `benchmarks/run_product.py` 复用产品目录和候选重排，明确输出 `ranking_scope=pre-evidence-assembly`、`dense_backend=local exact cosine reference`。该指标是候选排序而非最终证据段、OpenViking 线上索引、生成答案或真实用户任务准确率；与完整产品链路的差距不可忽略。
- 报告中的问题已在当前候选上作定向复测：假 provider、重排段、预算、大小限制和哈希回读均通过本轮对应检查；这不代表真实模型或生产验收。原报告按原样另存，以下各项保留当初发现过程。

## 已复现问题与状态

1. **P1：生成截断被当成成功；已针对假 provider 复测修复。** 初次探针用 MockTransport 返回 `finish_reason=length/content_filter/tool_calls` 及 `[DONE]`。原实现均正常结束，并可把 `partial [E1]` 判为结构有效；这不证明内容完整。主任务修改 `evidence_rag/answer.py` 后，同一探针对三种非 `stop` 原因均抛出不完整错误。证据：[修前](fake-api-probe-before-fix.json)、[修后](fake-api-probe-after-fix.json)、[探针](fake_api_probe.py)。修后只证明这三种构造响应，没有真实 provider 验收。

2. **P1：流式无效引用被保存并报告 ready；已针对假 provider 复测修复。** 在假的检索 bundle 和强制放行条件下，同一 `[E999]` 回答原普通 chat 返回 `insufficient_evidence`，SSE 则在验证前保存原文，发 `verification=invalid` 后仍发 `completed(status=ready)`。修后 SSE 发 `answer_rejected(discard_pending_answer=true)`、最终 `insufficient_evidence`，临时会话没有保存该回答。客户端已经收到的 delta 仍是 pending，客户端必须正确处理拒绝事件；探针未覆盖真实检索门控和生产前端。证据同上。

3. **P1：文档重排分与最后引用段落错配；已修复并定向复测。** `CrossEncoderReranker.rank` 聚合文档中各段的最高分（`evidence_rag/retrieval.py:63`），而 `RetrievalService.retrieve` 之后另用词法 `best_part` 选展示段（`evidence_rag/retrieval.py:198`）。在临时库中，假重排器给包含相关第 2 段的文档 10 分，最终证据却是无关的第 1 段，仍标 10 分、`ready`、`degraded=false`。因此分数/阈值并不能证明展示段支持问题。建议保留具体获分段落身份并使选段、分数、引用位置一致，再用独立样本复核；不要用该探针声称真实模型精度。证据：[rerank-part-probe.json](rerank-part-probe.json)、[探针](rerank_part_probe.py)。

4. **P2：tokenizer 实测分支缺少生成模型窗口及输出余量的联动检查；已修复并定向复测。** `generation_allowed` 对 `state=measured` 直接放行（`evidence_rag/api.py:139`），而生成器可不设置最大输出。假设测得 prompt 为 5,000 tokens、实际模型窗口为 4,096 且输出上限为空时，函数仍返回 true；这个假设不是实际模型测量，但证明门控没有接收窗口信息。建议将已核实的生成模型窗口、完整聊天模板开销和输出余量共同检查；不能把近似 byte 模式写成形式化硬上界。证据：[resource-probe.json](resource-probe.json)、[探针](resource_probe.py)。

5. **P2：导入大小限制在读取之后；已修复并定向复测。** 管理员接口先 `read_bytes()` 再检查 50 MB（`evidence_rag/api.py:279`）；CLI 输入适配器同样先整体读取再检查 100 MB（`evidence_rag/ingest.py:6`）。测试用 60,000,000 字节稀疏临时文件，并在实际读取前注入异常，观察到读取已尝试而非先返回 413；预备运行还实际读取该 60 MB 后得到 413。更大的文件可能先造成内存压力。接口需管理员身份和受限导入目录，这是可用性风险，不是未认证越权。建议打开文件后先检查可信文件描述符大小、按上限流式读取并二次校验。证据：[resource-probe.json](resource-probe.json)。

6. **P2：激活后数据库内容变更未在取证时重验哈希；已修复并定向复测。** 在隔离数据库中合法 stage/activate 后模拟底层数据损坏，`RetrievalService` 的指定 document_id 查询返回修改后的正文及原哈希，二者不一致；`status=ready`，因未配密集检索和重排而 `degraded=true`。这需要数据库写权限或实际存储损坏，不是远程认证绕过。建议对回读的 unit/part 实际文本重新计算哈希，不一致时拒绝作为引用证据并记录错误。证据：[catalog-integrity-probe.json](catalog-integrity-probe.json)、[探针](catalog_integrity_probe.py)。

## 测量及安全边界

- `rank_candidates` 共用不等于在线返回证据一致：重排段错配直接影响候选与证据层的关系；当前产品基准报告也明确声明不覆盖该证据组装及生成层。不要将检索 Recall/nDCG、服务健康、HTTP 200 或结构引用验证称为语义正确率。
- 当前通用语料按部署约定为认证用户共享，README 明确要求不放私有文件；本次未证明多租户文档级隔离。数据库损坏探针不能外推为未经授权用户可写数据库。
- 修复只经人工构造响应和临时数据库复测，尚需真实 provider/客户端验收。相关源码已变化，需重新冻结／记录代码与运行身份，不能把不同代码版本的测试分数混作同一组。
- 观察到的源文件可变：修后 `answer.py` SHA-256 `02581702ad1aafd593d737f4177b5f7aa3a54056612b0b5cd059d15d1ab818e1`、`api.py` `a7925dabf355121148b1a536534749b6adaadb1965152e735a9574e5a3899082`；修前探针未记录当时文件哈希，故只作为带输入输出的行为记录，不冒充固定提交的复现。评审产生的所有探针仅在本目录，不改业务代码。

## 2026-11-06 修复后独立复测

- 重排器明确返回第 2 段身份时，实际交付第 2 段且分数为 10；只返回文档分数而无段落身份时，返回段落的分数清空并标记 `reranker_part_unavailable`、`degraded=true`。详见 [修前](rerank-part-probe-before-fix.json)、[修后](rerank_after_probe.json) 和 [探针](rerank_after_probe.py)。这是人为构造的评分，不是实际模型表现。
- 模型窗口 4096、framing 256、输出预留 1024、最大输出 512 的假设下，prompt 5000 被拒、2000 被放行；未设置输出上限被拒。110 MB 稀疏测试文件在 HTTP 和 CLI 两条路径均在打开源文件前拒绝。详见 [修前](resource-probe-before-fix.json)、[修后](resource_after_probe.json) 和 [探针](resource_after_probe.py)。这里只证明构造值及所列大文件，不能证明未知 tokenizer 或服务端模板的界限。
- 激活后修改临时库 unit 正文，回读明确拒绝 `catalog unit hash mismatch`；另运行的项目测试覆盖 unit、part、document 哈希和 generic locator 四种损坏。详见 [修前](catalog-integrity-probe-before-fix.json)、[修后](catalog_after_probe.json) 和 [探针](catalog_after_probe.py)。这不是远程用户能够修改数据库的证明。
- 同一假 provider 复测 `length`、`content_filter` 和 `tool_calls`，均被识别为不完整；非法引用流式结果标记拒绝且未存储。详见 [最终复测](fake-api-probe-final.json)。
- 项目新加的 `tests/v2/test_final_integrity.py` 本次在独立环境 **14 项通过**，记录 [日志](final-integrity.log) 与 [JUnit](final-integrity-junit.xml)。未运行模型、OV、GPU 或 benchmark；真实 provider、最终答案语义和线上行为仍未验收。
- 每份保留的修前／修后 JSON 与当前源码哈希见 [retest-manifest.json](retest-manifest.json)。原始冻结矩阵的源码哈希不得用本次修后身份替换。

本公开副本保留发现时定位与源码hash；行号随修复可能移动，精确旧排序源码另存source-snapshots。此处的探针均为临时数据/假provider，不运行真实模型。
