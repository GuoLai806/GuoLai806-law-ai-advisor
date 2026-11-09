# RAG选型、成绩与已知缺口

本表是当前决定索引；与[用户版](PLAN-FOR-YOU.md)、[AI版](PLAN-FOR-AI.md)一致。原始备选、参数、失败和统计放在各仓库选型文档，按链接复核。

|决定|起因与为什么选|已知代价／确定程度|详细依据|
|---|---|---|---|
|通用核心+法律兼容|用户改通用方向，避免业务规则污染一般检索|法律资料仍需来源审核；不重建纯法律人工题库前置|[general决定](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/general-evidence-rag-20261106/TECHNOLOGY-DECISIONS.md)|
|BGE-M3+BM25+RRF+Qwen重排|语义/精确词互补，排名融合避不同分数域混合|Qwen更慢；32题模型差区间跨0，不称全局最优|[模型和完整参考检索](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/generic-benchmark-20261106/RESULTS.md)|
|真实OV0.5.3|用户指定；参考exact-cosine分数不是实际服务体验|固定资源scope/同版本原文；长文覆盖另验|[真实OV选型](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/openviking-user-flow-20261106/TECHNOLOGY-DECISIONS.md)|
|一次grounded_json+字面摘录检查|300题暴露NEI/引用问题；减少需要猜测的输出结构|可拒绝错摘录，非语义事实验证；产品text默认保留|[quality决定](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/openviking-quality-20261107/TECHNOLOGY-DECISIONS.md)|
|完全空输出至多一次重试|真实云模型空输出，必须保留每尝试/用量|其他错误不靠修JSON/隐式重試掩盖|[验证历史](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/openviking-user-flow-20261106/VALIDATION-HISTORY.md)|
|开发6/正式30|用户缩到原测试量十分之一|固定类别均衡和hash；小样本不作前沿结论|[30题协议](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/openviking-quality-20261107/REPRODUCTION.md)|
|连接复用+按需预览|借鉴bvio记忆入口，减少重复连接、便于阅读|warm find局部改善；完整答案仍检索，未证明整体提速|[exploration结果](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/rag-exploration-20261107/RESULTS.md)|
|loopback无令牌+最终答案/计时分区|用户页面要求；秘密留服务端，状态清楚|仅显式本机身份；草稿等待SSE完成|[页面依据](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/blob/1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1/docs/development/rag-workspace-clarity-20261107/PLAN-FOR-AI.md)|

## 结果版本与缺口



|口径|固定代码或阶段|结果|应保留的限定|
|---|---|---|---|
|共享排序+exact-cosine参考完整300|generic-benchmark阶段/8f433fb|nDCG .794175、Recall .935889、300执行0故障|预算前排名，非完整OV用户路径|
|真实OV用户300|7231a1fc00a20f5b311e083e7fcd2d5ed266dddb|nDCG .794201、Recall .935889、标签212/300、流程269/300|31失败全保留；NEI58/112；跨主机睡眠等待不能当生产延迟|
|30策略配对|561ec65292218541b380ff3963d21fb6115e6778|21→22正确、27→29流程、自定义引用F1 .5614→.7143、NEI4→6/10|整体区间跨0；字面引用核验不是独立语义判断|
|同配置直接/探索两组30|8da92ed阶段，0.2.2安装|均22正确/29流程，user_total P50 40.59→53.55s|预览方便阅读但未改善这批正确数量；阶段耗时含义不同|
|连接池warm loopback30pair|exploration C1|find中位24.7279→21.3095ms，30pair结果一致|网络find局部证据，非完整问答提速|
|最新页面工程/UI|c191866/0.2.3，source_map9140b021…f7144|451 Python、10Node；安装/API/hash/CUA开发题1134通过|没有重跑30/300；一次28.09s不作性能benchmark；.pth依赖复用明确|

质量指标定义在各自METHODOLOGY/RESULTS，不能把自定义文档+标签F1写成SciFact句子rationale榜单，不能将300题或旧回归数量套到最新代码hash。


证据定义见各阶段RESULTS/METHODOLOGY。真实300保留31失败，NEI58/112，预算后证据Recall .826611。30策略总体差95%区间[-10,+16.667]百分点；引用是自定义文档ID＋标签F1。预览qid1110墙1155.93秒、mono119.73秒差1036.21秒原因未知，未删题扣异常；部分usage缺失不据此称成本下降。

待办是中文/长文目标域与dense覆盖、完整后台队列、准确摘要名称、生产验收。已有同步ingest-jobs，字面摘录检查非语义支持判定。固定工程参数与32题候选对照不证明全局最优；本轮未跑模型。
