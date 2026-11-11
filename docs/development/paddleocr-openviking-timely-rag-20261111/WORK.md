# 需求方案工作记录

状态：2026-11-11 完成只读需求调查和开发规格草案，未实现业务、未安装PaddleOCR、未运行OCR/模型/数据库写入/服务重启。用户要求先使用“需求”做方案，开发待确认。

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

推荐直接搭真实时间线；事件日优先；正式验收时间有效性优先；中国大陆范围；现行与历史版本需来源/内容核验。工具未提供交互弹窗，故未将默认伪装为用户已确认。真正未决的是公开服务法律范围与法律人员复核门槛；它们不阻碍技术fixture，但影响正式权威发布。

## 交付物

[用户版](PLAN-FOR-YOU.md)、[AI执行版](PLAN-FOR-AI.md)、[调研证据与选型](RESEARCH-AND-DECISIONS.md)。AI版含R1-R10、任务T1-T7、验收、依赖、回滚及授权边界。计划提交本轮文档到现有可写分支并远端回读；不合并main、不改可见性、不发布产品。
