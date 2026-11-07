# 可使用结果与回退

代码、451项Python检查、10项Node检查、独立复审、wheel构建、仓库外安装，以及真实OpenViking网页问答/预览/原文/停止/改题均完成。用户本人接受尚待体验。当前GitHub状态：**等待本轮普通提交推送及远端回读**，完成后更新本记录。

正常入口：[本机RAG资料问答](http://127.0.0.1:19447/)，无需填写令牌。API绑定loopback19447，显式EVIDENCE_RAG_LOCAL_DEMO_USER=public-demo，AUTH已有非管理员普通身份；其他Bearer接口沿用。OV19443和embedding19444继续原服务，未重载、未重建索引。当前体验使用grounded_json+grounded_evidence，产品默认text未改。

## 后续开发与公开复现

参考[前轮复现](../rag-exploration-20261107/REPRODUCTION.md)准备真实OV、公开SciFact语料、embedding、reranker与模型配置。远端Agent自行提供私有运行配置，凭据不会归档。

```sh
python -m pip install '.[dev]'
python -m pytest -q tests/v2
node --test tests/v2/workspace.test.mjs
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/rag-wheels
```

构建环境需有setuptools>=75。这些命令复现工程检查；不等于实际模型/网页验收，模型依赖与公开权重需按前轮说明准备并固定revision。本轮本机安装用新venv经.pth复用现有模型环境，未做干净依赖解析。

服务启动参数增加local_demo_user字段，须映射到AUTH已有非管理员。启动benchmarks.serve_user_flow关闭proxy_headers；直接uvicorn启动也必须绑定127.0.0.1且使用--no-proxy-headers，不依赖转发头判断本机。无此字段仍需Bearer，不能把演示无令牌模式推广到公网。

本机任务状态保留install-venv、wheels-v2、tested-source-v2.tar.gz、catalog-before.sqlite3与管理脚本manage_run.py；脚本以既有安全stdin传私有配置。真实catalog复用前轮任务库。GUI原始在本机outputs/rag-workspace-clarity-20261107/ui，服务/会话/检查原件在work/dev/rag-workspace-clarity-20261107；这些不是远端运行依赖。[公开身份](evidence/source-identity.json)与[原件hash](evidence/raw-records-manifest.json)可核对。

回退：先核实19447监听进程命令、归属和端口，停止并确认退出；用保留的0.2.2 wheel与原配置启动同端口可恢复旧页面。仅需关闭本机直接访问时移除LOCAL_DEMO_USER并重启API即可恢复Bearer边界；新页面无令牌表单，此时可用原脚本接口。OV/embedding保持现状。catalog数据无迁移，通常无需恢复；真需恢复备份时先停止API，核验一致备份和目标库，禁止覆盖在写库。凭据和旧安装原件留本机，不发布。

授权覆盖普通开发、必要安装/验证/推送；不包含公网部署、force、合并PR、改仓库可见性或其他项目开发。GCode/SuperAgent/MyAgent/简历继续[TODO](../rag-exploration-20261107/TODO.md)。
