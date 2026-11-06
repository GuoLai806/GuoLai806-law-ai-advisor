# 最终代码复审记录

本次只读复审的范围为基线 `8f433fb2e0dd1ac171c7c797de571c2e8840eb3b` 至 HEAD `7231a1fc00a20f5b311e083e7fcd2d5ed266dddb`，加工作区非流式 API 修复、新增分析/审计/选择工具及对应测试。未访问运行服务、调用云模型、加载模型、修改仓库或重跑正式 benchmark。本文是独占新增的本机评审记录。

## 结论

本次检查没有发现新的确定 P1/P2 缺陷。先前 P2 非流式 `/api/v2/chat` 的重试用量丢失已修复。修复仅位于该非流式 handler，正式 SSE 测试对应的冻结候选仍为 7231a1f，不能把修复后 API 文件哈希与正式冻结候选混称同一源码。

## 代码修复复审

`api.py` 现在按 attempt 保存每次调用最后一次累计 usage，再把已知调用相加。generation_attempt、generation_retry、model_identity 独立保存；成功回答、非法引用和异常回退都返回已知 accounting。

已报告全部调用用量时 `usage_source=provider`、`usage_complete=true`；部分调用缺报为 `provider_known_attempts`、false；尝试生成但没有任何用量为 `unavailable`；未调用生成器为 `not_generated`。缺报不补零。无 attempt 的旧 usage 归入默认 attempt1；仅文本 stream() provider 仍可完成回答，用量保持未知。

观察到专项 `chat-accounting-after-fix-v1.log` 为 10 passed，0.16 秒；其修复前 v2 为 10 failed，0.21 秒。测试覆盖两次已报累计、同调用累计覆盖、分别缺报首次/第二次、完全缺报、重试失败保留、非法引用保留、无生成、无 attempt legacy event、仅文本 stream 兼容。`regression-v15-all.log` 实际为 tests/v2 范围 162 passed、5 项 PyMuPDF 弃用警告，3.30 秒。不同范围的历史测试次数不能简单相加报告为 180。

## 正式评分与工具复审

本次用独立算式从全部 300 行重建混淆矩阵、标签一致率、三类 macroF1 和引用文档与标签 F1，结果与原始报告一致：31 失败仍在分母，212/300 标签一致（0.7066666667），macroF1 0.7395578465，引用文档与标签 F1 0.5357737105。没有以成功的 269 题作为质量分母。

分析输入 SHA256 与原始正式报告一致；按全部可用行独立重算四种耗时的 NumPy p50/p95，均一致。原始报告与分析保存的 NIST trec_eval 舍入核对通过。源排名消融的每个来源前 20 覆盖无缺口，300 行完整；它只说明同次请求的检索阶段差异，不说明完整答案 A/B 或其他系统效果。

增强审计从原始 SSE 独立解析首行标签、提取实际引用文档，并重建最后累计 usage、attempt/retry/model identity；失败预测保持空。原始 27 项通过记录应保留，不能在源码修复后改写其 source 身份。正式结果检索 nDCG@10 为 0.7942005100，Recall@20 为 0.9358888889；这些分数与标签质量、实际流程完成率分开，不能用检索高分推导全流程 SOTA。

## 复现与覆盖限制

当前 HEAD 工作区 API 已改变，`audit_user_flow.py` 严格核对当前运行源码哈希时，会对历史正式配置产生 source mismatch，这是正确行为。后续重审原始正式记录应使用 7231a1f 冻结源码 checkout，把交付的独立 tools 放入该 checkout 执行，保留工具自身 hash；不能在最终修复版直接执行后忽略 source mismatch。交付文档仍需更新正式状态并解释此版本边界。

本 agent 独立复审了主 agent 编写的非流式修复；增强 audit 和专项测试先前由本 agent 编写，其后续检查属于自查，不能称为另一个完全独立 reviewer 对这些工具的审查。复核算式没有复用 aggregate() 来重建质量指标，但不能取代专家对证据和答案的语义审核。

未验收生产并发、复杂 PDF/OCR/长文层级检索、多轮上下文或隐藏 SciFact sentence-rationale 榜单；未审定最终公开包脱敏、GitHub push 或浏览器实操结果。共享硬件和单次云模型运行的置信区间不能证明稳定行业领先。文档当前尚在收尾，最终应使用真实结果并维持上述限制。

## 内容快照

下列 SHA256 指向本次实际检查的内容；后续若改代码，应对受影响范围补审。

{
  "evidence_rag/api.py": "0a85423686feebff6d6cc9a9a47cf319a9f774f76fe051fefd4658451b23d4c4",
  "tests/v2/test_chat_usage_accounting.py": "9a5384dbb8da12769ffce76a6e63527e7cf48e0a1488525c79c38bdc75f5524e",
  "tests/v2/test_user_flow_audit.py": "249af3d578deff2fe4d638799e8c7efd7a663bedb43839d69b535feee9c712d7",
  "tools/audit_user_flow.py": "a6941f6ab70b7727e207ea7b23187c6aa3a525caf9f9ca1ed3e16dbb4d53b9b7",
  "tools/analyze_user_flow.py": "8d1ed2e65a2c6d6ea5dbe642cd07099f7766dcda540b5424607fe28f579b21cb",
  "tools/select_user_flow.py": "55182b9240d28f56f0a9ad13ead395f2835d8dc67914004021ad64cfafb08b09",
  "tools/compare_user_flow_validation.py": "0d86aec628abf8f4c8c0009ed8ca4437ff2d8404d3ebee9953455b76dcf67c8f"
}
