# 页面清晰度与本机直接访问实施依据

[用户版](PLAN-FOR-YOU.md)。基线 019d2a5f3fc9bb30b5c63369973f00b0bf1f8634；仓库 GuoLai806/GuoLai806-law-ai-advisor；分支 feat/versioned-legal-rag-20261105；开工工作区干净。

R1：删除令牌 UI 与浏览器 Bearer。本机显式启用 LOCAL_DEMO_USER=public-demo，须是 AUTH 中已有非管理员；直接 loopback socket、本机 Host、同源 Origin/Sec-Fetch-Site 才能无 Bearer。默认 401、错误 Bearer 不能 fallback、其他 owner 仍 403。启动关闭 proxy_headers，不依赖 X-Forwarded-*。

R2：先检索内容，再 AI 最终答案；资料可展开完整段落与同 revision/hash 原文。仅 completed ready 且无 error 展示最终答案；取消、重试、未完成不能当最终答案。中文结论标签与引用展开，后端 canonical 答案与会话不改。数据只用 textContent。

R3：performance.now 计提交到 evidence_ready/预览响应、evidence_ready 到流结束（生成与核验）、提交到结束（总等待）。失败/停止显示未完成，未知不伪造 0；预览不计生成，人工阅读不计再次问答。改题 abort 旧请求，epoch 拒旧响应；停止清理旧查找提示。

修改 api.py 身份与固定 mjs 路由、static/index.html 和 workspace.mjs、serve_user_flow 白名单、pyproject 0.2.3 与资产打包、专属 Python/Node 测试。benchmarks/e2e.py 将新增 mjs 纳入未来源码冻结身份；旧正式记录不改。检索/生成/grounding、模型配置和旧正式数据不改。沿用 FastAPI/httpx/DOM，无新产品依赖，Node 只测前端行为。

实际已按此顺序完成：实现、auth/API 与计时状态测试、独立只读复审修复、wheel 安装及模块/资产 hash、核实停止原 19447、同端口启动候选、CUA 问答/预览/原文/停止/改题及会话回读、记录和双版、普通提交推送逐文件回读。Git/进程/GUI 由主 Agent 单一操作者。复用 OV19443、embedding19444 和原 catalog 一致备份，不重载这两项、不重建索引。

验收：无令牌且本机可直接问答；三块结果、三段耗时、最终答案与引用清楚；同版本原文完整读取；失败、停止、未知不混称完成；错误身份/非本机/跨源仍拒绝；安装 API/HTML/mjs 与候选 hash 一致。只跑必要工程回归和单开发题 smoke，不视为正式质量提高。

本机保存原始响应、截图、失败、构建/安装日志；公开版本、配置摘要、数字、失败/hash，不含正文和凭据。安装 venv 用 .pth 明确复用模型依赖，不声称全新依赖解析。检查、安装、网页状态分别记录，用户本人验收待体验。

只部署 loopback19447，保留旧 wheel/安装/DB 备份。关闭 LOCAL_DEMO_USER 恢复 Bearer 边界，回旧 wheel/API 配置可回旧页面；停止前核对命令、端口、归属并等退出，未知写不重放。普通开发、安装、测试、推送已授权；不公网发布、不 force/合并、不改其他库和原正式评测。无实施待确认项；公网登录、中文质量、新正式性能评测范围外。

## 已执行状态与后续边界

代码候选为 0.2.3，source_map_sha256=9140b0215a98ae139b0fa2d299f3cef859621d557495dbb4038c877cb30f7144，完整文件身份见[evidence/source-identity.json](evidence/source-identity.json)。451 项 v2 Python 测试、10 项 Node 测试通过；P2 部分/空预览错误完成提示已关闭。wheel 已构建安装；新 venv 经 .pth 复用现有模型环境，不是干净依赖解析验收。

真实安装 API 无 Bearer 200、错误 Bearer/跨源/错误 Host 401；原 owner 隔离由12项专属鉴权测试覆盖。SciFact scifact_full_v3 5183 文档，真实 OpenViking0.5.3、BGE-M3、Qwen reranker 与 DeepSeek 配置前后身份一致。CUA 实际问答、预览、原文、正确停止、改题通过；canonical session 回读一致。第一次停止晚于已完成，保留为未验收通过尝试；浏览器取消不代表 OV 后端立即终止。详见[结果](RESULTS.md)与[复审](REVIEW.md)。

本轮实际问答只用开发题1134，不改变或重跑正式30题，不将单题秒表作为质量/前沿/提速证据。中文为UI标签；本轮没有改变模型生成语言。正式语义评测沿用前轮结果。公开证据仅版本/配置摘要、检查日志、原始hash与失败说明，原始网页正文/响应/会话、SQLite和凭据留本机。

共享Git、进程、GUI由主Agent串行完成，评审子任务已接收且停止写入。服务保留19447，OV19443与embedding19444未重启；回退、正常入口和公开复现见[交付](DELIVERY.md)。必需项已通过；用户本人接受待体验，不自动新增优化。GitHub同步记录更新在DELIVERY；不得只凭本地提交宣称同步成功。

代码、双版和公开证据已同步 GitHub，提交 [c191866](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/commit/c191866f4b4f7247fa9a63953bda398bd8f24a78)；32 个改动文件逐一远端回读，与本地内容一致。[回读记录](evidence/github-code-sync-v1.json)。
