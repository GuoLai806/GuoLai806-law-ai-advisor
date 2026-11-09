# RAG接手执行依据

[用户版](PLAN-FOR-YOU.md)与本页同决定；本项目当前入口。核对基线1ad62b83e8f2c2a57a4b602bfdc1bc7a5f57c9d1，分支 `feat/versioned-legal-rag-20261105`，main=8f433fb2e0dd1ac171c7c797de571c2e8840eb3b。本轮文档提交不改变代码身份。

## 目标与范围

使接手AI从HANDOFF找到历史问题、有效决定、现行源码/接口、每阶段证据和暂停边界。验收对应HISTORY/DECISIONS、current-state、既有文档索引和本次远端逐文件回读。用户授权普通文档提交/非force推送；本轮无业务代码、安装、重启、模型题、索引重建、部署或合并。

继续保留OpenViking、通用公开评测、后续开发6/正式30、独立公开实现和原始失败。原法律人工300题前置已被替换。SuperAgent、其他项目联动与进一步CU等不属本轮开发；更换AI不自动续跑旧WORK。

## 现行接口与模块

核心 `evidence_rag`，`legal_rag`薄兼容；default general。catalog.py维护document/unit/revision/parts/hash；retrieval.py统一BM25、OV、RRF及重排；answer.py/grounding.py管理生成和摘录结构；exploration.py只供资料预览，不能直接作为生成证据。api.py管理身份和会话，static/index.html与workspace.mjs显示最终状态。

POST `/api/v2/retrieve`、`/api/v2/explore`、`/api/v2/chat`、`/api/v2/chat/stream`；GET `/api/v2/units/{unit_id}?revision=...`、`/api/v2/sessions/{session_id}/turns`、`/api/v2/models`、`/health/live`、`/health/ready`。Query字段以api.py的Pydantic模型为准；session owner由服务器绑定，不能由请求自报。单位回读校验同revision/hash。当前生成门槛函数 `generation_allowed`，旧文档may_generate是历史名称。

POST202 `/api/v2/ingest-jobs`同步读取/解析/stage并持久jobs，GET状态存在；尚无完整后台队列。CLI ingest/activate和viking_index send/reconcile/validate导入完整revision，激活现有版本expected CAS；unknown先inspect/reconcile，不重放。API scope必须与活动revision一致。

实际OV0.5.3，vectors_only/no_split，BGE-M3 1024维，BM25+RRF+Qwen3-Reranker-4B。当前demo `grounded_json + grounded_evidence`；产品默认 `text + reranker_threshold`。正常一次生成，只有限定空输出最多重试一次、记录每尝试usage；字面摘录不是语义事实验证。

页面等completed ready、无error且SSE结束才展示最终答案，改题abort/epoch隔离，performance.now计阶段延迟。本机无Bearer只允许显式LOCAL_DEMO_USER普通身份、直接loopback与Host/Origin/Sec-Fetch约束，proxy_headers=False；错误Bearer不fallback，部署公网前需正式身份与TLS。

## 当前环境、依赖与恢复

[evidence/current-state.json](evidence/current-state.json)是2026-11-10快照：19447 ready/scifact_full_v3，19443 OV0.5.3，19444 BGE-M3/MPS；资料5,183标题摘要。11-09报告“服务停止”随后已被恢复。今天仅GET，不发模型题或重建数据。接手重查进程/端口/版本和在途请求。

Python3.12、公开模型固定revision、安全提供的云模型凭据与独立数据库目录是运行依赖；不从公共文档复制密钥。源码README/examples与各轮REPRODUCTION提供可移机命令。先[最后安装交付](../rag-workspace-clarity-20261107/DELIVERY.md)→[阅读索引](READ-FIRST.md)→对应复现→源码/测试，查清用户新目标再执行。

回退使用0.2.2及一致Catalog备份；先确认自己的API进程已停止再切，OV/embedding无需因UI回退重建；禁覆盖在写DB。关闭LOCAL_DEMO_USER恢复Bearer。文档错误用后继普通提交；本轮无产品迁移。

## 验证与待办

[DECISIONS](DECISIONS.md)集中保存成绩绑定、指标、失败、未知与选型代价；451Python+10Node是0.2.3工程结果，不覆盖旧300/30源码。长文完整dense覆盖、中文目标域、完整队列、生产并发未验收；当前“完整原文”在SciFact仅是摘要。接手AI待用户选方案，不把拟议规格或单开发题延迟写成正式收益。

发布白名单HANDOFF.md、README.md、docs/development/handoff-20261110；不带企业/Pi源码、私人聊天、Word、原GUI/响应、模型权重或DB。所有新增文件GitHub ref/tree/blob回读，业务文件身份不变；完整跨项目历史由私有归档管理，本公开入口自足。
