# Evidence RAG 执行依据

与 [用户版](PLAN-FOR-YOU.md) 为同一决定。范围：通用文档检索、可选legal profile、证据组装、受控生成、API及可复现评测；不承诺专业法律质量、生产多租户隔离、跨语种最佳效果或生成主张语义正确。

## 仓库和接口

仓库 GuoLai806/GuoLai806-law-ai-advisor；起点689f2a5bf4b853dfe93ac26802970f8996f993c8，开发分支feat/versioned-legal-rag-20261105。核心 evidence_rag/{catalog,retrieval,budget,answer,api,ingest,viking_index,cli}.py；legal_rag为兼容包装，不维护第二套排序。benchmarks/从源码运行；模型/语料/缓存均在仓库外。

JSONL每行 `_id/title/text`，附加source_url/metadata可选。通用默认provided来源只表示提供者给定，不意味着官方核实；法律来源须额外审核。Catalog.stage_documents建立不可变revision、units/parts/FTS；activate使用expected revision CAS。OpenViking URI包含corpus+revision；send先journal，reconcile核对L2内容哈希，validate再核对向量可见性。已有unknown条目不重新派发。

HTTP：GET /health/live、/health/ready；POST /api/v2/retrieve、/api/v2/chat、/api/v2/chat/stream；GET /api/v2/units/{id}、/api/v2/sessions/{id}；管理员POST /api/v2/ingest-jobs。Bearer token由部署者映射主体，会话首次绑定主体。当前语料按约定对所有已认证用户共享。

SSE依次含retrieval_started/evidence_ready/answer_delta（pending）/usage/verification/completed。非法引用发answer_rejected(discard_pending_answer=true)，completed为insufficient_evidence，不保存无效回答；异常流发error并要求丢弃pending。模型必须finish_reason=stop且有完成标记，length/filter/tool_calls或缺结尾拒绝。客户端须处理拒绝/错误，不能将收到的delta当最终已核验答案。

## 依赖与运行

Python>=3.12，pip install .；可选.[dev]和.[models]。constraints-runtime.txt是本机干净wheel安装版本快照，不是带哈希跨平台锁。模型revision见retrieval.py及benchmarks/run.py；OpenViking独立固定0.5.3，本机实际验证使用vectors_only、content_only、1024维BGE-M3端点；资源/模型路径和token均由私有配置指定。

先ingest入独立db，核对解析和来源；只需词法浏览可activate并显式查看degraded。完整向量流程按unit执行 python -m evidence_rag.viking_index send/reconcile/validate，固定相同db/revision/url/scope；每个send后人工/脚本核对reconcile结果，失败停止，不删除journal换路径重试。新索引验证后activate --expected切换旧revision。源不一致创建新revision，不改已激活内容。

API设置EVIDENCE_RAG_DB、AUTH（token到主体的JSON映射）、OV_URL、OV_SCOPE、RERANKER=qwen、RERANK_DEVICE/DTYPE、MODEL_URL/NAME/KEY。TOKENIZER及TOKENIZER_REVISION必须匹配生成服务的模板；MODEL_CONTEXT_WINDOW、FRAMING_RESERVE、OUTPUT_RESERVE和MAX_OUTPUT_TOKENS显式设置，完整prompt+余量不得超窗，一般问答另需针对目标分布标定RERANK_THRESHOLD。阈值未校准时仅展示资料，不照抄本数据集数值当通用校准。

试验字节模式还需ALLOW_CONSERVATIVE_BYTES=1及请求budget_mode=conservative_bytes，只允许general；返回exact_tokens=false。该试验不使严格ready变真。STORE_CONVERSATIONS仅显式设1才落对话；公开语料服务不要导入私有资料。容器配置使用仓库根context和loopback端口，仅config检查通过，未启动daemon或生产服务。

## 评测、验收和顺序

用户已授权实现、测试、提交/推送公开独立代码；不推企业/Pi私有实现、凭据、用户GUI原文、权重或原始语料，不改MyAgent。本轮无需额外权限问题。正式生产认证/TLS和真实法律来源审核未完成。

先对官方语料校验下载MD5/SHA和qrels，再冻结32 validation；v2 2x2模型用于候选比较。因段落归属/完整性修复，v3重新冻结代码与配置；逐篇5183条输入等价、旧分数CPU回放和向量cache复制有独立审计，不把旧成绩重标成v3。v3选定配置重新真实推理32题和词法baseline，排序与v2一致。随后为长测数据留存补逐题fsync日志和提前输出冲突检查，冻结v4后再次32题真实验证和词法baseline，零失败后各自selection冻结，再完整test300；不以test调参。模型顺序加载释放、MPS float16/batch2，各路top40、RRF60、cap60；记录每题排名、分数、来源、失败、用量、缓存、预热和环境，空结果/失败进分母。TREC -c交叉核对，结果仅pre-evidence候选排序。

源码回归105项通过、独立复现修复、全新wheel安装/CLI通过。真实OV/BGE/HTTP小样本与真实云模型SSE另存：首次脚本import失败未发模型；修环境后HTTP200、实际usage89+150=239。此为受控synthetic、近似字节预算、引用结构通过且claim_support=not_verified，不是问答质量测试。

最终manifest应锁源码、报告、selection、语料与qrels、模型revision、wheel和远程commit；完整test未出前不得填最终指标。失败保留原文件和解释，输出唯一文件；不同版本的结果明确分开。回退使用旧活动revision和兼容代码，保留新revision/journal/原始证据；未知OV写入只核对不重放，数据库备份遵循SQLite一致性/WAL规则。

活动revision与OV_SCOPE必须一起切换：停止新请求并drain旧实例，activate --expected后以新scope启动应用并验证；回退使用旧指针与旧scope，不承诺零停机。

发布顺序：先保留已推开发分支，完整验收及证据提交后重新核对origin/main；可安全快进时更新主线，保留开发分支，不force覆盖他人工作。实际远程状态以最终交付记录为准。

长文范围：当前通用dense单元为完整输入文档，超过模型窗口须上游准备有来源的短单元，不能把OpenViking估算截断当分块。SciFact全部序列化文本通过默认4096估算不变检查，但任意长PDF/复杂版面/OCR未完成质量验收。参考examples输入策略说明。

导入语义：每次ingest形成完整revision快照，新增文件不自动合并活动语料；应准备合并后的完整输入再创建新版本。完整命令见examples/README。

最终验收更新：完整官方300题、零执行故障；nDCG@10 0.794175（产品词法0.668854）、Recall@20 0.935889（0.836556）、MRR@10 0.762501（0.635099），NIST -c -q逐题与总体核对通过，冻结source前后相同。真实选定Qwen重排、公共BGE端点、既有OV合成索引和云模型SSE通过小样本互操作；预算为显式近似字节模式，claim_support仍not_verified。正式全文排名及journal见generic-benchmark-20261106/RESULTS.md。
