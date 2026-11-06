# 开发、失败、检查和实际使用验收历史

本页对应[用户版](PLAN-FOR-YOU.md)与[执行版](PLAN-FOR-AI.md)，被测实现561ec65。保留真实失败和早期检查，不用测试数量替代系统质量。

## 实现与独立审阅

可选grounded_json/grounded_evidence、一般场景资料不足判断、摘录校验、双时钟、30题冻结和审计已实现。开发发现并修复：科学ANSWER不能靠正文骗取标签分；异常后仍保留grounding/usage；完整开发报告与双selection冻结必需；从原始UTC/ns重建计时，不能只信报告flag。

最终完整regression-v4=310 passed/5warnings，5.50秒。独立需求/记录审阅144项及7边界通过；另一计时/冻结范围76项通过，均核对源码固定，不累加为独立唯一测试总数。包含MockTransport边界，但只证明实现与记录逻辑；正式真实后端与安装/UI另验。

## 开发6题不删历史

| 记录 | 适用范围 | 标签 / 流程 / 失败 |
| --- | --- | --- |
| candidate-dev6-v1 | 评分/失败元数据修复前 | 4/6；6/6流程；无失败 |
| candidate-dev6-v2 | 同策略早期复核、评分修复前 | 4/6；5/6流程；1 invalid_quote |
| candidate-dev6-v3 | 最终源码，用于正式冻结 | 4/6；6/6流程；无失败；audit通过 |

三个版本模型、提示和索引参数没有按正式题调整；v3在确定评测代码缺陷修复后另名复核，不覆盖v1/v2，也不选择历史最高成绩。v1/v2是历史报告，不能当它们已通过最终源码冻结审计；公开导出保留实际记录与各自source/config。

## 正式30×2已经结束

两selection预先冻结，baseline/candidate各一次，无重跑，无正式结果调参。完整报告各30、全部失败保留；旧21/30、27/30流程、3gate；新22/30、29/30流程、1invalid_quote。两材料审计、模型文件字节、source/runtime、gold、raw/journal、原文/会话、usage和双时钟通过。TREC复核通过。后端before/after身份一致，raw after额外诊断load_seconds保留。

本轮primary正确率差区间跨0，NEI区间跨0，引用F1区间碰0。完成此次实际系统验收不等于达到高精度长期目标。全部原始SSE本机保留；公开5报告78行，数字/排序/失败/用量/hash一致，正文和路径脱敏。

## 构建、安装和启动观察

wheel0.2.1构建成功，hash `cbdcff20aca3c70300fa2207fbe3fbd4dc9d2579e0c688edfb597ae93c532376`。完整联网下载依赖两次因速度终止，CLI/import过早检查失败原日志保留，不能叫完整干净安装成功。

最终独立环境用--no-deps安装本项目wheel，并通过显式.pth复用已验model/runtime依赖。四项安装模块与HTML身份通过、CLI --help通过；运行目录在仓库外、PYTHONPATH清除，实际evidence_rag来自安装包。未验任意全新机器从零依赖下载。

安装服务启动早期ready一次503(answer_model_health)，保存该观察；随后真实API和最后ready成功。正式19446已停止后才启动19447，避免额外模型实例干扰正式计时。

## 真实安装API及浏览器

安装后固定开发qid1134，真实SSE交付正常文字/摘录校验/structural verification、trace=OpenViking、4条同版本原文hash、完成会话回读通过；请求/原始事件本机保存，独立摘要公开。不进正式30。

CUA浏览器：填写演示令牌与固定开发题，先验证提问后pre-completion停止，显示“已停止”“未完成的草稿已撤回”；另一次正常完整问答显示SUPPORT、中文解释和原文摘录，再点击E1“查看原文”看到同篇内容。截图/AX各3份和hash保留，仅摘要公开。功能测试是一个固定样本和取消边界，不是广泛GUI可靠性benchmark。

最后读取ready/model identity/served HTML，候选runtime及wheel/源码HTML一致。只绑定本机loopback的19447入口保留供用户查看；OV19443与embedding19444保留作为依赖，正式19446已停。Catalog以SQLite backup一致备份到本机任务状态；不推数据库/模型权重/配置密钥。

## 交付检查

权威双版、RESULTS、TECHNOLOGY、RESUME、REPRODUCTION、本文、evidence及README入口同组更新，仓库内相对路径可读。最终证据与文档由独立review再核对，从需求覆盖和数据可信度分别报告；普通commit/push后读取远端提交及文件内容确认。远端交付确认保存在本机，不在正文预先冒称成功。

生产发布、并发、多领域/中文/长文、人工逐句支持、LoRA或企业指标不在本次验收范围。代码/工程检查/安装实际入口已验证；新机完整依赖下载及“稳定业界前沿”明确未验或未证明。
