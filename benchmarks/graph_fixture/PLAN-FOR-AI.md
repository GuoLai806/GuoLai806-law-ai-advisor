# AI 版：真实依赖图草稿

对应 [用户版](PLAN-FOR-YOU.md) · [资料与验证说明](README.md) · 2026-11-18。

目标是提供可复核公开软件依赖事实，支持主任务的开发试跑。事实源为 27 个 npm 版本端点，精简源快照保留版本、依赖字段、许可证、gitHead 和原始响应 SHA-256；`docs.jsonl` 另记录快照 SHA-256、来源 URL 与 `/dependencies`，`edges.jsonl` 对每条精确版本边给出 JSON Pointer。许可声明为 MIT、ISC、BSD-3-Clause。只有快照自身哈希可完全离线重算；原始响应哈希需要现场下载字节完全一致的响应验证，registry 元数据更新会改变响应字节。

数据契约：文档一行一个 JSON 对象，字段为 `_id,title,text,source_url,metadata`；关系 `source,predicate,target,evidence_id,locator,spec`。仅纳入目标节点也在语料中的精确版本 `dependencies`，这是人为选定的子图；不得把有范围的依赖当精确版本，也不得推断缺边为无依赖。不要给图索引声明 `coverage_complete=true`。版本是包自身不可混淆的身份，不是经过验证的日期有效区间；待取得日期依据前不要给日期查询标注有效或无界支持。

题目包含 12 条跨文档单路径、6 条版本对照、6 条普通单文档、6 条歧义/证据不足；`gold_path` 是一种有效路径，存在多个可行路径时评分器应接受有证据的其他答案；版本比较和否定题的语义需人工复核。题目尚未盲标；本集已被作者构造和检查，作开发集，不用于正式质量阈值。必要后续工作：建立能证明足够覆盖的历史固定版本图和日期证据，核验完整闭包金标，独立审核并冻结新的正式题集，接入 Catalog 与 OpenViking、实跑质量和速度。所有结果需记录真实后端、权限、配置和证据预算。

本地复现：`python3 benchmarks/graph_fixture/build.py`；`python3 benchmarks/graph_fixture/validate.py`。前者只读已保存快照，`--fetch` 会更新源快照，仅供明确需要重新取数时使用。无部署及回退操作；主任务负责集成、独立审阅、提交与推送。本数据模块不修改其他工作区文件或 Git 索引。
