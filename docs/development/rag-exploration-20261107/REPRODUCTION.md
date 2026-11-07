# 复现连接与探索全流程对照

[用户说明](PLAN-FOR-YOU.md) · [执行依据](PLAN-FOR-AI.md) · [源码身份](SOURCE-IDENTITY.json)。新机按相同配置重新测量；云模型别名不保证未来输出相同。公开脱敏报告不能替代原始SSE审计。

## 数据和服务

沿用[上一轮数据/索引准备](../openviking-user-flow-20261106/REPRODUCTION.md)和[6/30题身份规则](../openviking-quality-20261107/REPRODUCTION.md)。本轮实际索引是OpenViking0.5.3，SciFact5183篇、scifact_full_v3，标题换行正文序列化已逐篇核验并标记dense_validated；没有重新导入或拿矩阵文件冒充服务。固定清单在benchmarks/manifests/scifact-short30-20261107.json，hash为8ee8fd71d89777b93ef27e062fffd81ddb8f419b7c476eddc16bf2a08c6d4690。

本机BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181；Qwen3-Reranker-4B revision22e683669bc0f0bd69640a1354a6d0aebcfeede5；MPS float16、batch8、同生成tokenizer、4条/6000 tokens完整证据。模型deepseek-v4-pro、thinking enabled/low、temperature0、输出8192、主timeout60秒；仅完全空输出允许disabled/45秒/2048的一次重试。配置按公开config文件，不调整正式错题。

私有服务JSON从公开config的service_configuration补上db/auth/ov_url/tokenizer/model_url/model_key；保存仓库外0600文件。OV_REUSE_CLIENT=1、EXPLORE_ENABLED=1。两组都ANSWER_FORMAT=grounded_json和GENERATION_GATE=grounded_evidence；顶层user_flow_strategy给runner使用，不改变服务。启动：

```sh
python -m benchmarks.serve_user_flow --port 19447 < "$RAG_RUN/private-service.json"
```

只绑定127.0.0.1。核对/health/ready、经鉴权/api/v2/models与后端health；必须看到实际OV scope、正确模型/提示hash/预算与模式。令牌仅由stdin或私有环境送入，不打印配置/凭据，不修改全局代理。

## T1 同30题实际网络

```sh
python -m benchmarks.run_openviking_connections \
  --url http://127.0.0.1:19443 \
  --scope viking://resources/rag/scifact/scifact_full_v3 \
  --manifest benchmarks/manifests/scifact-short30-20261107.json \
  --data-dir "$RAG_DATA" --output "$RAG_RUN/connections.json" \
  < "$RAG_RUN/private-headers.json"
```

stdin为{"headers":{}}或本机实际私有鉴权header。runner使用生产adapter的reuse_client=False/True，30题AB/BA交替、串行、无自动重试；两条预热单列。保存完整find rows/双时钟/失败/源码前后和health身份。成功pair才有配对时差；失败仍在60请求/30pair分母。输出和journal存在即拒绝覆盖。mock只用于行为测试。

## T2 固定两个实际用户路径

先用candidate config跑validation6：/explore→打开首条同版本原文→实际/chat/stream→所有引用原文→会话回读。直接路径正式没有前两个动作。max_evidence_items4，include_retrieval_trace开启用于审计；网页默认不带trace。两组都实际生成并核对摘录，不能拿组件或参考后端结果当此对照。

```sh
python -m benchmarks.run_user_flow --url http://127.0.0.1:19447 \
  --data-dir "$RAG_DATA" --manifest benchmarks/manifests/scifact-short30-20261107.json \
  --claims "$RAG_RUN/claims_train.jsonl" --config "$RAG_RUN/explore.config.json" \
  --output "$RAG_RUN/dev6.json" --raw-dir "$RAG_RUN/raw" --split validation \
  < "$RAG_RUN/private-auth.json"
```

stdin为{"token":"本机令牌"}。先独立audit完整6题，不因质量不足重跑寻优。冻结两组：

```sh
python tools/freeze_short_run.py --comparison-strategy exploration \
  --manifest benchmarks/manifests/scifact-short30-20261107.json \
  --development-report "$RAG_RUN/dev6.json" --development-config "$RAG_RUN/explore.config.json" \
  --baseline-config "$RAG_RUN/direct.config.json" --candidate-config "$RAG_RUN/explore.config.json" \
  --baseline-test-output "$RAG_RUN/direct30.json" --candidate-test-output "$RAG_RUN/explore30.json" \
  --baseline-selection "$RAG_RUN/direct.selection.json" --candidate-selection "$RAG_RUN/explore.selection.json"
```

配置差异仅顶层user_flow_strategy：direct和explore_then_answer；其余值严格一致。公开selection本机路径已脱敏，复现须重新生成本机pair；不能删除source/pair/development检查绕过冻结。

分别使用对应config、output、selection跑一次test，claims换成claims_dev.jsonl；先direct30，再explore30，单用户串行。每个输出是新路径，失败、错误/缺摘录、timeout和不完整usage保留。正式不调参。

## 独立审计和比较

每个report调用tools/audit_user_flow.py，提供--report/--config/--manifest/--claims/--qrels/--raw-dir/--output，正式加--selection；validation用train.tsv，正式test.tsv。审计重建原始SSE、失败、引用、usage和双时钟；探索另核对实际HTTP响应、首条原文hash和同version。失败还没收到后端时明确计为未观察，不声称验证过真实OV。

```sh
python tools/compare_short_user_flow.py --baseline "$RAG_RUN/direct30.json" \
  --candidate "$RAG_RUN/explore30.json" --raw-dir "$RAG_RUN/raw" --output "$RAG_RUN/paired.json"
```

配对所有30题，失败计错；保存Wilson、10000次配对bootstrap和exact McNemar。每题时间从用户动作开始，探索包含前置等待，service_total只计回答服务器。双时钟可能受系统时间调整/睡眠影响，异常不丢题；不推断速度的单因素因果关系。预览字节是JSON body实测；evidence_ready大小是解析后JSON重编码，不是wire或总流量。

## 安装与网页

构建wheel0.2.2，在仓库外新venv安装，核对实际导入模块和HTML hash、路径及服务启动cwd；本轮.pth复用已装模型依赖，公开记录明确这一限制。使用网页先看资料→打开原文→回答，再走停止和修改问题取消旧探索；完成答案再查同会话保存。UI测试在正式benchmark结束后执行，避免并发影响计时。

保留旧安装和DB一致备份；OV_REUSE_CLIENT=0回逐次连接，EXPLORE_ENABLED=0关闭预览；默认text保持不变。仅停止本轮登记PID，不改企业/Pi/GCode服务。原始记录本机保存，公开只发布数字/hash/脱敏证据，原始预览与原文不能作为旧发布工具aux。其他项目见[待办](TODO.md)。
