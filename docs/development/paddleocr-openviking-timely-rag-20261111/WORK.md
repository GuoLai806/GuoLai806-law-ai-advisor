# 开发工作记录

状态：2026-11-11 隔离实现与验证完成，等待提交/推送和后续独立质量评测。历史“只调研、未实现”记录保留在下方，已被新指令替代。


## 2026-11-11 实施确认

用户明确选择：1) 自动抓取版本；2) RAG自动检索并按时间信息工作；3) 当前范围仅民法典；4) 生成扫描/图像版本测试PaddleOCR；5) 不引入额外法律发布审批，只做技术自动检索与来源/结构核验。需求已确认，进入实现；原“本轮不直接开发”已由用户新指令替代。

当前实现基线：T1-T6 已完成隔离闭环，T7 进行提交/推送。已新增 `temporal.py`、时间过滤/候选有效性约束、法律 PDF 行清理与结构字段、结构上下文嵌入序列化；测试覆盖事件日优先、过期/未来条文排除、未核验来源、法律结构、全量 1260 条索引和时间 benchmark。
## 已确认用户要求

TECH-1 使用PaddleOCR解析民法典，技术价值写高精度OCR和数据清理。TECH-2 使用OpenViking做结构化索引。TECH-3 使用Timely RAG做时间维度评分。TECH-4 先给可开发文档，本轮不直接开发。延续RAG项目：保留真实OpenViking、通用公开评测基线、证据与失败、现有公开/私有边界。

## 核查事实

- RAG仓库分支/HEAD：`feat/versioned-legal-rag-20261105` / `0d14a09773c54c9432475a4e357bbe893304d97b`；工作区开始时clean。
- 民法典PDF：64页原生文字，约116,848字符，无嵌入页图；正文1,260条已在既有catalog；当前manifest来源pending、有效期空。
- `catalog.py`解析目/条/款/项与relations；`retrieval.py`已读取event_date但未时间过滤/打分；`viking_index.py`现有whole-unit ingest和journal。
- OpenViking 0.5.3实际接入、BGE-M3、BM25/RRF/Qwen为当前RAG基线；OpenViking本地源码有目录层级、L0/L1/L2与time decay API，但不等同法律有效期。
- TimelyRAG upstream commit `5213a0c1743f23fd52a258005c3887aaf7e8cd0f`；官方pipeline/论文的四时间信号、二阶段重排和合成韩语数据已核查。论文引用仅作为外部方法依据，未迁移为中文法律成绩。
- 现有120题民法典集是AI编撰、全部需人工审阅、非盲测；仅极少数含显式时间表述，不能证明时间能力。

## 建议默认与未决

推荐直接搭真实时间线；事件日优先；正式验收时间有效性优先；中国大陆范围；现行与历史版本需来源/内容核验。工具未提供交互弹窗，故未将默认伪装为用户已确认。工作区根目录保留本轮导航双版本，项目内保留完整[用户版](PLAN-FOR-YOU.md)与[AI版](PLAN-FOR-AI.md)。真正未决的是公开服务法律范围与法律人员复核门槛；它们不阻碍技术fixture，但影响正式权威发布。

## 交付物

[用户版](PLAN-FOR-YOU.md)、[AI执行版](PLAN-FOR-AI.md)、[调研证据与选型](RESEARCH-AND-DECISIONS.md)。AI版含R1-R10、任务T1-T7、验收、依赖、回滚及授权边界。计划提交本轮文档到现有可写分支并远端回读；不合并main、不改可见性、不发布产品。

文档自检完成：用户/AI双版、调研依据、WORK 4 个文件；本地相对链接检查通过。未安装或运行OCR，未改模型/索引/数据库/服务，业务代码未变。

## 2026-11-11 最终实施状态

- 后端全量回归最终结果：503 passed、5 warnings；原始 `evidence/pytest-v2-final.log`。
- 前端 Node：11 passed；原始 `evidence/node-v2-final.log`。
- wheel：`evidence_rag-0.3.0.dev0-py3-none-any.whl`，126,583 bytes，SHA-256 `589ee363e708029395d3a534cdc04ff58ced6aca26dcdda68faa845865ecec08`；独立环境导入与 CLI 帮助通过。
- PaddleOCR 3.7.0 / PaddlePaddle 3.3.1 实际 64 页：1260/1260 条边界，1259/1260 字面一致，唯一复核条保留；不是零错误。
- OpenViking 0.5.3：1260/1260 send+内容回读、1260/1260 可见性验证、匹配排名均 1；索引 201.49 秒，验证 30.28 秒。
- 时间正式集 30 题：RRF nDCG@10 .487698、Recall@20 .5、stale leakage 1.0；RRF+Timely .987698 / 1.0 / 0.0。作者构造定向集，不宣称通用前沿。
- 真实 API：施行前 0 条、insufficient；生效期内 8 条有效证据；explore 首条完整正文 hash 回读一致；chat 无模型时 evidence-only。
- 未完成：Qwen/BGE cross-encoder 权重、生成模型、独立人工评测、多版本历史法、官方可信 PDF 正文下载、production 并发与容量。
- 开发记录：起步 HEAD `87c64c4c2d5ca10fe95e71e1bec36193a3ba8312`，分支 `feat/versioned-legal-rag-20261105`；本轮代码与证据尚未提交时不得声称远端完成。
