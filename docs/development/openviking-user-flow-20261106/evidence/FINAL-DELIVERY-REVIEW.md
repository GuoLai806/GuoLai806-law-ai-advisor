# 最终交付需求与文档复审

本次只读复审 canonical `docs/development/openviking-user-flow-20261106/` 下全部九份主文档及公开证据副本，与本机正式原始 report、最终验证 pair、公开行业调查和测试日志核对。未改仓库、运行模型、访问服务或重跑正式题。仅新增此评审记录。

## 结论

未发现新的确定 P1/P2 数值、脱敏或需求一致性问题。用户版与 AI 版使用同一冻结选择、正式结果、失败和后续改进决定；用户版能独立阅读，AI 版给出接口、依赖、执行顺序、验收、部署/回退、版本和授权边界。简历句子以真实检索指标与工程范围为依据，没有把检索召回当回答正确率、把阶段排名提升当完整答案 A/B，或宣称完整 RAG SOTA。

发现的三项事实措辞已经通知主 agent：RESULTS 曾仍称全项目测试进行中，已更新为实际180通过；OpenViking原文读取分工已明确为索引验收，用户原文GET由Catalog/SQLite读；wheel-v9独立安装仍在收尾，安装成功状态由主 agent核对后补齐，不在本次评审中认定完成。

## 数字与范围复核

正式5,183资料、300题；31失败、269完整流程；212/300标签一致，macroF1=.7395578465，doc+labelF1=.5357737105；NEI58/112、SUPPORT101/124、CONTRADICT53/64。17题门槛仅给资料、未调用生成；283题尝试生成，293次attempt，10题重试，269题有usage、259题attempt报告齐全。公开和正文没有把所选证据都说成实际送云。172/188完整gold文档覆盖只在正类题，所选证据平均Recall和前20检索Recall分开。

正式nDCG@10=.7942005100、Recall@20=.9358888889。BM25=.6688975974，差+.1253029126、相对18.73%；RRF重排前=.7042731827，差+.0899273273。配对区间与原始统计一致，明确是同次请求排名阶段对比；不推导完整答案增益。所有失败保留分母，开发320记录是重复32题，不说成320独立题；11报告合计620记录，也未说成620独立问题。

延迟明确披露macOS单调时钟及Sleep：UTC25073秒，逐题秒表和10378.623392秒，Sleep交集14756秒；p50/p95没有被改写成日历用户等待或线上容量。未把睡眠与连接错误当已证因果，也未据此筛失败。全文与简历说明NEI和引用弱、结构引用不等于语义正确。

行业表14行与公开 industry-survey.json 的14条同task/revision记录一致，调查配置38个，没有把不同revision、翻译/Nano/LMEB混榜。NV-Embed-v2和stella等nDCG与Recall均披露；混合检索+重排和单embedding组成不同，外部只是有来源参照，未冒称全行业全局排名或同硬件性能胜出。

## 公开证据与入口

独立逐行核对11个公开report、620rows的关键数字、标签、失败、排名、source/session flags、usage与raw hash，未发现差异；11份公开config与原始字节一致，公开report hash与脱敏登记一致。公开JSON递归检查answer/request/query/text/events/auth/api_key/model_key字段及/Users私人路径，无发现。全部本地Markdown链接目标存在。

这项扫描不能代替全形式秘密检测，也不证明每句话受到文献支持。原始SSE和论文正文留本机，公开只能复算数字，不能独立重做原始流或原文一致性审计；文档已经说明。原27项增强审计在7231冻结源码、非流式修复前进行；修后10专项、tests/v2 162与tests全180是后续代码证据，不能把最终HEAD说成重跑300题。

## 仍由主 agent最终核对的事项

截至本次观察，wheel-v9安装首次依赖下载失败保留，retry1安装尚待确认；REPRODUCTION末句仍提到wheel-v9安装入口作为范围依据，应明确待最终记录或在实测成功后再写完成。当前评审不认定独立安装完成，也不认定GitHub推送、远端SHA、最终DB/OV备份和专用服务清理已经完成。该状态提示是交付收尾要求，不是新业务许可请求。

本 agent先前编写了audit增强和专项测试，对其复查属于作者自查；此次RESULTS/行业表和主 agent双PLAN/选型/简历的审查独立于这些文档作者。行业原始来源联网复现、专家语义裁决、中文/跨领域/长文、持续唤醒延迟、高并发和完整答案强基线不在已验证范围。文档如实保留这些限制。

## 文档内容快照

{
  "PLAN-FOR-YOU.md": "5be35856f4e7c3574d9228cd9da2192430da32ca47b76722dbc2fadb74d47121",
  "PLAN-FOR-AI.md": "a213baf58c1992476c72c6a5bd431ddb421bc8a29c79de2743b639949456b74e",
  "RESULTS.md": "7e6484a483148b1b2daafd363d477f2b7a2c299c5b9571ba29fe93ca6c9565c3",
  "INDUSTRY-COMPARISON.md": "641f9b976bbfd12ac54422f8db80a70a50cc4a1bde6c01a98d021449403de66e",
  "TECHNOLOGY-DECISIONS.md": "675e013130326995d077d9d4affa09145d95627ecbaa473e2525a8db68803a34",
  "RESUME-REWRITE.md": "8099ea45eb5dd4eb135e6baae2ffd42b3696e6f1337b0047c14e61d9b5318236",
  "METHODOLOGY.md": "91460a37b32bd80e9ddd5ec51cb6c66d16dfea2bc4911b6e3c4c7cd3c6fdcf84",
  "REPRODUCTION.md": "a1344351f9f51a601b8ecbbc3a5259726d78435e03daa2b7e3cae453f21a4de3",
  "VALIDATION-HISTORY.md": "08b977a21f8ae6ce0b3f8ed0ab3f4c3d9192e2688a84dce7afe7ea2e3b17654d"
}
