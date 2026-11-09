# 通用RAG状态交接

[用户版](PLAN-FOR-YOU.md)，日期2026-11-09。[快照](evidence/status-snapshot.json)固定远端、本地、服务和历史证据hash。本轮只做进度文档，不授权新业务、merge/main更新、force、部署、重启或模型评测。

基线7358f5031f00054ece5240ed034b53ad70491042，feat/versioned-legal-rag-20261105；main8f433fb2e0dd1ac171c7c797de571c2e8840eb3b，ahead17、behind0、localclean。core0.2.3源码9140b0215a98ae139b0fa2d299f3cef859621d557495dbb4038c877cb30f7144，历史451v2+10Node，wheel/安装/UI已验收；.pth模型依赖复用非干净解析。300题与两批30题各自绑定版本/配置，不套到最新源码。今天19447/19443/19444拒绝连接，原因未知，未启动或重测。

范围：本目录双版/evidence快照与README入口；不得改evidence_rag、benchmarks、tests、正式记录。验收是三个阶段的指标及失败限制清楚、源码/安装/今日运行区分、main与branch分明、两版同决定、GitHub逐文件回读。沿用FastAPI/OV/模型依赖，无新依赖或接口。本次无构建/质量测试需求。顺序：快照→文档→链接/隐私/事实核对→白名单commit→nonforce推送→ref/commit/tree/blob回读→本机receipt。回读未知先核查，不重放写。

当前待办：按已有管理脚本恢复演示三服务（需要演示时）、选择目标资料和中文长文未见评测、过滤/队列生命周期、生产认证/并发/平台验证、main整合。UI“原文”在SciFact就是标题/摘要措辞待修；本次无代码修改。SuperAgent及其他项目暂停方案仍有效，不并行启动。无新的必须用户确认项；未来需求选择再弹窗。文档回退后继提交更正；业务回退沿用[DELIVERY](../rag-workspace-clarity-20261107/DELIVERY.md)。
