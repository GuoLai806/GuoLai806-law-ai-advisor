# 可追溯的依赖关系检索

这是 Evidence RAG 通用文档模式的可选功能。OpenViking 继续负责现有原文和语义索引，SQLite 的图表保存经过校验的关系，Python 程序逐步查可达的依赖。普通检索保持现有入口。第一种可查询关系为 `depends_on`，不限制遍历深度。

## 本地使用

1. 按[现有示例](../../../examples/README.md)导入、验证并激活通用语料。构图需要已审核（`approved`）的通用语料及可回读段落。资料只有文件级来源时先补原文定位、校验证据和关系日期。
2. OpenViking v0.5.3 的 `ov compile` 可导出实体 Markdown 和 `relations.jsonl`。启用 Compile 需要 OpenViking 的运行环境；[版本化官方示例](https://github.com/volcengine/OpenViking/blob/v0.5.3/docs/en/context-compilation/03-knowledge-graph.md)说明其设置。把编译输出放到现有检索 scope 外，导出目录可用以下命令转换为待核对输入。

```sh
python -m pip install PyYAML
python -m evidence_rag.graph_index compile-candidates --directory ./kg-export --revision demo-r1 --output ./graph-pending.json
python -m evidence_rag.graph_index --help
```

3. 用提供的模板建立已经审核的 `graph-reviewed.json`。候选不自动变成可检索证据。运行 `build`、`validate`、`activate`，各子命令参数见 `--help`。切换构建时指定期望中的旧图 ID。旧构建保留，供同语料版本回退。
4. 在已授权的服务上设置 `EVIDENCE_RAG_GRAPH_ENABLED=1`，可选配置 `EVIDENCE_RAG_GRAPH_BUDGET_MS`（默认 750 毫秒，独立于目录/重排阶段）。打开普通问答页，写问题并点「关系检索」。起点无法唯一确认时，可填完整名称或版本，问题日期可用日期控件指定。查看结果的「查看原文」验证具体支持，同一结果的分页绑定图、语料版本、起点和日期。

页面在每次返回证据后显示「检索过程」。方框按查询范围、关键词、语义召回、合并与重排、图谱和证据的实际参与情况排列，法律查询有日期筛选记录时也会显示。点方框可查候选排名和分数、图谱关系和原文证据，手机端可横滑。盒内耗时来自服务端本次记录，右上角是服务端检索耗时；下方原有「本次耗时」仍显示浏览器等待。阶段没有独立计时时省略该时间，检索尚未返回时隐藏回放。

普通问答与关系检索从现有 `evidence_ready` 事件取数据，「只检索资料」从预览响应取数据。此界面使用既有可选追踪字段，沿用原文端点的版本与哈希核对。

### BGE 重排的输入检查

使用 `EVIDENCE_RAG_RERANKER=bge` 和显式 `EVIDENCE_RAG_RERANK_DEVICE=mps` 时，服务在预测实际使用的分词批次内检查每条输入长度。超过模型窗口的段落会触发现有重排降级。其余配置沿用逐条检查。可设置 `EVIDENCE_RAG_RERANK_WINDOW_CHECK=legacy` 并重启回退，`/api/v2/models` 返回实际采用的模式。

用同一语料和模型比较两种模式时，运行仓库的 `benchmarks/compare_reranker_windows.py`。脚本对真实 HTTP 端点交错请求，记录候选顺序、分数、原文哈希、图谱边、耗时和失败。两端须使用同一模型、设备、精度、批大小和 OpenViking 数据快照。

`POST /api/v2/retrieve`、`/api/v2/chat` 和 `/api/v2/chat/stream` 支持 `retrieval_mode=graph_dependency` 和可选 `graph_seed`。不指定模式按普通检索处理。图未建或关闭时，可转回现有普通问答。对证据不足或版本不明的图结果，单独展示已查到的路径；此时不会生成声称找全的 AI 回答。队列耗尽仅表示当前图的可达范围查完。分页显示额度、遍历运行预算和生成选用的证据预算各自独立。

## 复现开发样本

[Express 开发语料](../../../benchmarks/graph_fixture/README.md)可用来重现真实 OpenViking 与图检索。它使用公开固定版本元数据，尚未解析 npm 范围约束，因此图显示覆盖不完整。发布日期表示这些版本的清单已经公开，不能用来推断实际部署中的依赖。

```sh
python -m benchmarks.graph_fixture.prepare_runtime --db ./fresh-demo.sqlite3 --revision npm-graph-dev-r1 --ov-url http://127.0.0.1:19343 --report ./index-report.json
# 设置本地 Evidence RAG 实例的相同 DB 和 scope 后，从 API 运行两条路径。
python -m benchmarks.graph_fixture.collect_runtime --url http://127.0.0.1:19445 --output ./graph-runtime-results.json --rounds 9
```

复现实验只对经过明确配置的 loopback 测试服务运行。`prepare_runtime` 要求新数据库，索引失败时检查 journal 和服务端实际写入，再决定如何继续。原始索引不受图开关影响，关闭开关并重启即可回到旧的普通查询。
