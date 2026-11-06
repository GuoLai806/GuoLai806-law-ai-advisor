# 复现这轮 6 题检查和 30 题对照

先看[大白话说明](PLAN-FOR-YOU.md)和[执行依据](PLAN-FOR-AI.md)。这一页只解释怎样复现这轮小样本流程；不要求重新跑历史 300 题。

## 数据和运行条件

使用公开 SciFact：BEIR 的 5,183 篇资料及 train/test 检索标注，原 SciFact 的 claims_train/claims_dev 判断标签。数据下载、许可和原文核验沿用[上一轮复现文档](../openviking-user-flow-20261106/REPRODUCTION.md)的准备、索引步骤，停止在正式模型评测之前。本轮继续用真实 OpenViking 0.5.3 的 `scifact_full_v3` scope；新机器可以重新导入并逐篇核验，同机可以复用已经核验的索引。

新清单是 `benchmarks/manifests/scifact-short30-20261107.json`。开发 6 题来自 train，每类 2 题；正式 30 题来自 test，每类 10 题。种子 `20261107`，每类按 SHA256(seed + query_id) 排序抽样。清单 hash 为 `8ee8fd71d89777b93ef27e062fffd81ddb8f419b7c476eddc16bf2a08c6d4690`。全量 qrels 和 claims 仍需取得并核对；不能只准备抽出的题，更不能按预测结果改抽样。

```sh
python tools/prepare_user_flow_subset.py \
  --manifest benchmarks/manifests/scifact-validation.json \
  --data-dir "$RAG_DATA" \
  --claims-train "$RAG_RUN/claims_train.jsonl" \
  --claims-test "$RAG_RUN/claims_dev.jsonl" \
  --output "$RAG_RUN/scifact-short30.json"
```

本机实测为 macOS/Apple M5 Pro、MPS float16，串行一题一个用户请求。BGE-M3 embedding revision `5617a9f61b028005a4858fdac845db406aefb181`，Qwen3-Reranker-4B revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`；重排 batch 8，证据最多 4 项/6,000 tokens。生成使用已配置的 `deepseek-v4-pro` 云别名，thinking enabled/low，temperature 0，输出 8,192，主超时 60 秒；仅完全空输出允许 disabled/45 秒/2,048 的一次重试。别名和 temperature 0 不保证未来云端模型字节或回答相同。

## 两套服务配置

旧配置为 `answer_format=text`、`generation_gate=reranker_threshold`、threshold 0；新配置为 `answer_format=grounded_json`、`generation_gate=grounded_evidence`。其余检索、证据、预算和模型主参数相同；新格式会拼接 JSON/摘录契约，属于联合策略，不能把结果全归因一个因素。产品默认仍保留旧配置。

公开 config 只含模型设置、源码/数据 hash 和预期 runtime identity。服务还需本机私有的 DB、tokenizer 目录、OpenViking URL、模型 URL/key 和 auth 映射。把这些字段加到私有 JSON，用 0600 权限保存在仓库外，通过 stdin 启动，不把完整配置或密钥写入命令参数、报告或 Git。

```sh
caffeinate -i -m python -m benchmarks.serve_user_flow --port 19446 < "$RAG_RUN/candidate-private.json"
```

启动后检查 `/health/ready` 与经过鉴权的 `/api/v2/models`。必须看到真实 OpenViking scope、正确重排/tokenizer/model/prompt hash 和 generation mode。配置 identity 不符时 runner 在发题前拒绝，不要删检查强行跑。不要改全局代理；需要直连时只对当前下载、推送命令调整连接方式。

## 先开发 6 题，再冻结两个正式运行

从仓库根目录运行 benchmark。令牌只通过 stdin 送给 runner；下面假定私有 `auth-input.json` 仅包含 `{"token":"你的本机令牌"}`，保存在仓库外。

```sh
caffeinate -i -m python -m benchmarks.run_user_flow \
  --url http://127.0.0.1:19446 --data-dir "$RAG_DATA" \
  --manifest "$RAG_RUN/scifact-short30.json" \
  --claims "$RAG_RUN/claims_train.jsonl" \
  --config "$RAG_RUN/development.config.json" \
  --output "$RAG_RUN/development.json" --raw-dir "$RAG_RUN/raw" \
  --split validation < "$RAG_RUN/auth-input.json"
```

先审计完整 6 题 report/journal/raw。模型失败会返回非零，但 report 仍完整保存；不能删失败题。只有确认代码 bug 并修复时，允许用新运行名复核同 6 题，不重复试到分数满意。

```sh
python tools/freeze_short_run.py \
  --manifest "$RAG_RUN/scifact-short30.json" \
  --development-report "$RAG_RUN/development.json" \
  --development-config "$RAG_RUN/development.config.json" \
  --baseline-config "$RAG_RUN/baseline.config.json" \
  --candidate-config "$RAG_RUN/candidate.config.json" \
  --baseline-test-output "$RAG_RUN/baseline.json" \
  --candidate-test-output "$RAG_RUN/candidate.json" \
  --baseline-selection "$RAG_RUN/baseline.selection.json" \
  --candidate-selection "$RAG_RUN/candidate.selection.json"
```

candidate 正式 config 必须与最终开发 config 完全相同。baseline 沿用历史 text/threshold 0 配置，没有单独挑选开发分数。工具同时锁定两套配置、开发记录、样本、源码、互指 selection 和正式输出路径，任何正式 report/journal 已存在都不能重新冻结。公开 selection 的本机路径已脱敏；复现时重新生成本机 pair，不能直接拿脱敏文件绕过校验。

分别启动对应服务后，各跑一次下面的 test 命令，替换 `MODE=baseline` 或 `candidate`。输出与冻结的路径必须完全相同。服务 token 也必须与对应私有配置一致。

```sh
caffeinate -i -m python -m benchmarks.run_user_flow \
  --url http://127.0.0.1:19446 --data-dir "$RAG_DATA" \
  --manifest "$RAG_RUN/scifact-short30.json" \
  --claims "$RAG_RUN/claims_dev.jsonl" \
  --config "$RAG_RUN/$MODE.config.json" \
  --output "$RAG_RUN/$MODE.json" --raw-dir "$RAG_RUN/raw" \
  --split test --selection "$RAG_RUN/$MODE.selection.json" \
  < "$RAG_RUN/auth-input.json"
```

## 审计、统计和实际产品验收

对每份正式 report 运行 `tools/audit_user_flow.py`，提供 report/config/manifest/claims/qrels/raw-dir/selection/output；正式小样本 selection 为必需项。审计核对源码与后端身份、原始事件、失败、标签、引用、usage、会话/原文回读、UTC/ns、异常和分位数。模型错误不能靠审计变成正确答案；审计通过只表示记录内部一致。

```sh
python tools/compare_short_user_flow.py \
  --baseline "$RAG_RUN/baseline.json" --candidate "$RAG_RUN/candidate.json" \
  --raw-dir "$RAG_RUN/raw" --output "$RAG_RUN/paired.json"
```

配对统计包含全部 30 题、失败计错、Wilson 区间、配对 bootstrap、exact McNemar，以及首字/完整流/原文与会话回读的真实日历时间。冷暖缓存、运行顺序和网络可能影响速度。原文摘录存在检查与语义正确分开；NEI 无摘录允许交付，并单独计数。已有公开小样本不证明未见泛化或行业前沿。

最后从服务网页提问，核对可读答案和摘录、打开同版本原文，再验证停止会撤回未完成草稿。构建 wheel 后在仓库外安装并核对实际导入文件 hash，再启动安装产物验证同样的真实入口。保存日志、版本、失败和安装依赖复用范围。仅服务健康、组件 mock 或测试数量不能代替真实路径。

新模式可切回 `text + reranker_threshold` 回退，不改已有索引/DB schema。保留原始本机答案和 SSE、全部失败及一致 DB 备份；公开仓库用 `tools/publish_short_user_flow.py` 导出脱敏数字、原始 hash 和配置，再独立核对发布材料。
