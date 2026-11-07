# 页面清晰度实际结果

本轮两项用户要求已实现并经过安装后的真实系统路径验证：去掉令牌输入；按检索资料、AI最终答案和延迟展示。[用户版](PLAN-FOR-YOU.md)与[执行版](PLAN-FOR-AI.md)同组决定。本轮没有重跑正式质量评测。

|要求|实现|验收与证据|
|---|---|---|
|删除令牌输入|HTML无输入框，JS不发送Bearer；显式本机普通身份|真实API无Bearer200；错误Bearer、跨源、错误Host401；[API记录](evidence/actual-api-v1.json)，12项专属鉴权检查|
|先资料再最终答案|独立三块；草稿缓存到completed ready无error且流结束|真实网页4条资料及最终答案；会话回读一致；[网页摘要](evidence/ui-acceptance.json)|
|延迟清楚|performance.now三段计时，预览不计生成、改题独立计时|真实问答18.81+9.28=28.09秒；真实预览17.94秒且未请求生成|
|保留真实后端与数据|未改模型/检索/生成门槛/正式记录|OpenViking、embedding、generation身份前后一致，5183文档scifact_full_v3|
|原文与引用可核对|同revision/hash原文；引用摘录独立展开|实际问答与预览原文成功打开；字面摘录存在不证明语义正确|

## 工程与安装

- 451 项 Python v2 回归通过，运行5.11秒；pytest汇总5项警告，另有异步fixture默认配置提示和退出时SWIG弃用提示，原日志保留。[最后候选日志](evidence/pytest-v2.log)。
- 10 项 Node 检查通过；此前9项的版本保留，新增部分/空预览检查后变为10。[最后前端日志](evidence/node-v2.log)。
- 独立复审发现P2部分检索误显示正常完成，修复复审通过，无阻塞发现。测试数量仅说明对应工程检查。
- 0.2.3 wheel构建、安装成功；API、HTML、mjs在仓库外安装位置且hash与候选一致。[安装身份](evidence/installed-source-v1.json)。新venv经.pth复用已有模型依赖，不证明全新环境依赖安装。
- 最后源码摘要9140b0215a98ae139b0fa2d299f3cef859621d557495dbb4038c877cb30f7144，wheel SHA256 36f240e5583930b5c4daa478a34ddf6de1b9ed676a55ea8459f67812819dcc45。[完整身份](evidence/source-identity.json)。早期构建候选保留，未安装；最终构建见[build-v2](evidence/build-v2.log)。

## 实际用户路径及失败

实际平台macOS Apple Silicon、Python3.12；loopback API19447为安装0.2.3，OV19443为OpenViking0.5.3，embedding19444为BGE-M3；Qwen3重排、DeepSeek-v4-pro与grounded_json/grounded_evidence配置继续沿用。[配置基线](evidence/backend-before.json)包含具体revision与预留配置。ready只表示配置/组件准备状态，实际生成由网页路径验证。

只使用开发题1134：4条检索资料、最终答案、引用摘录及完整原文均可见，canonical会话回读与保存的回答一致。一次问答28.09秒，不作为正式benchmark样本。随后只检索预览17.94秒、未请求生成，原文成功打开。

第一次“停止”点击发生在24.33秒的回答已经完成之后，观察到两个正常completed turn；这次尝试不能记为停止通过。后来在检索期立即停止：0.37秒“已停止”，资料未完成、生成尚未开始，没有旧查找提示。预览中改题后旧结果未覆盖新输入，计时已清空。此处只验证浏览器取消与状态隔离，没有验证OV后台工作立即终止。

独立复审曾误打开默认ignored空SQLite，主流程核实5个业务表全0、无句柄、143360字节，可恢复移至本机任务状态。真实SciFact库未受影响；备份integrity_check=ok。[空库记录](evidence/incidental-review-db.json)，[备份摘要](evidence/backup-before.json)。

## 证据与结论边界

本轮没有准确率或吞吐结论，不改变前轮正式30题对照的限制。UI中文标签不改变模型正文语言；正文仍可能英文。网络延迟、排序、生成、引用核验一起包含在浏览器等待中，人工阅读时间不计入再次请求。错误与未知不伪造为零。

原始UI文本、截图、会话与服务记录保留本机，公开日志仅替换本机路径，失败和警告不删；[原始hash清单](evidence/raw-records-manifest.json)可对应本机原件。用户本人满意度待体验。代码、检查、构建、安装、正常入口的状态分别见[DELIVERY](DELIVERY.md)。
