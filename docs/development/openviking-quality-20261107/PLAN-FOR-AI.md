# OpenViking回答质量与30题对照：执行依据

与[用户版](PLAN-FOR-YOU.md)使用同一决定。关联仓库GuoLai806/GuoLai806-law-ai-advisor，分支feat/versioned-legal-rag-20261105，基线b62887d686894f50811ba54d0c4ed08f4b6e8fc9。用户已确认继续开发，正式benchmark由300缩至30；既有开发、必要验证、普通commit/push授权沿用。当前terra不在宿主可用模型列表，子任务沿用父模型，不强设Astra。

## 需求、范围和边界

| ID | 明确要求 | 实现与验收证据 |
| --- | --- | --- |
| R1 | 真实OpenViking继续使用 | 复用已验5183资料/scifact_full_v3 scope，实际find；每题trace、原文/会话回读 |
| R2 | 测试约为原来十分之一 | 新manifest validation6(train每类2)、short_evaluation30(test每类10)，模型调用前锁样本；旧manifest仍完整300 |
| R3 | 改善NEI与引用质量 | 可选单调用grounded_json响应契约，摘录substring检查、最少充分来源、NEI条件提示，失败不改判NEI；与text旧配置同30对照 |
| R4 | 全流程真实等待 | UTC start/end、wall和monotonic双时钟、原始事件wall时间、时钟差；抑制本轮闲置睡眠 |
| R5 | 原始证据/选型理由 | 保存每题、全部失败、usage、数据/source/config/model身份；实测与预期分开 |
| R6 | 交付可用结果和长期材料 | 代码/测试/构建/安装/真实API/UI/复审分别验收；两版文档与脱敏资料同步同GitHub分支，核对远端SHA及文件 |

不修改历史300报告、gold、索引、模型权重或原Word。仅RAG；GCode/MyAgent/Pi/企业实现/生产配置不改，不合并PR、不force、不改可见性。不创建独立聊天。所有共享Git、服务、GUI、DB与主PLAN写入由主agent统一操作。

## 接口与模块决定

- benchmarks/run_user_flow.py：旧字段保留。新manifest的short_evaluation明确绑定30 IDs、labels、seed、claims与qrels；test selection新增manifest_sha256。记录每题UTC与wall总/首字/stream，raw事件received_wall_ms；保存monotonic旧字段与clock_info，不从差距筛题。旧manifest不含short_evaluation时仍test300。
- tools/prepare_user_flow_subset.py：CLI接受原manifest、claims train/dev、输出；不覆盖。每label按SHA256(seed20261107+query_id)选前N，validation每类2、正式每类10，保存确定性规则与来源hash。三类均衡，标注public small paired experiment，非隐藏未见集。
- evidence_rag/grounding.py：解析严格JSON {decision:SUPPORT|CONTRADICT|NOT_ENOUGH_INFO|ANSWER,answer:string,evidence:[{id,quote}]}。拒重复key/非法字段/未知ID/引用不一致/伪造quote。只做空白规范化的title+text substring，quote12..1200字符、最多4个证据项。SUPPORT/CONTRADICT/ANSWER至少1quote；NEI可0quote，渲染参考资料明确不构成支持证据。无效响应硬失败，不能转NEI或再叫模型修JSON。
- answer.py：OpenAIStream新增answer_format=text|grounded_json，默认text。messages_for支持同格式提示，预算和实际消息一致，legal保持text规则。grounded模式在同一次上游SSE缓冲最多64KiB，及时转发模型/usage/attempt；stop/done后核对，yield grounding_validation摘要再单个正常正文delta。不伪造逐字。沿用旧空回答最多1retry，格式错误且已有raw content不retry。
- api.py/serve_user_flow.py：operator ANSWER_FORMAT、GENERATION_GATE。默认text+reranker_threshold完全兼容；grounded_evidence仅general+grounded_json，仍要求ready、budget、完整证据和无degraded，允许负相关性logit进入模型判断。引用摘录校验代替裸相关性阈值作交付条件，但semantic_support=model_assessed_not_independently_verified，不能称独立语义核验。legal不放宽。JSON与SSE都保留grounding_validation、安全失败类型与attempt usage；runtime_identity记录格式、gate及prompt hash。
- static/index.html：正常文字显示，grounding事件仅展示自然状态，不显示内部JSON；停止撤回、原文/完成会话维持。
- tools/freeze_short_run.py及审计/分析：只在正式30 journal不存在时写配置选择，绑定source/config/manifest，保存development6身份；正式两模式配置在发题前冻结。新audit支持明确subset，不能取消source/gold检查。统计同30配对，失败留分母，CI及小样本/历史已见限制明确。

## 依赖、数据与部署

沿用Python3.12/macOS arm64、现有model-venv及已固定Qwen/BGE/DeepSeek tokenizer，不升级依赖。OpenViking0.5.3、BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181、Qwen reranker22e683669bc0f0bd69640a1354a6d0aebcfeede5。DeepSeek Pro thinking enabled/low、8192输出、60秒；仅空输出disabled重试45秒/2048。证据4/6000、窗口32768/framing512/output8192。云alias保存model/fingerprint，不称固定云权重。

本机新任务目录outputs/openviking-quality-20261107、work/dev/openviking-quality-20261107。Catalog由旧一致备份复制到新DB；复用旧已验OV scope和磁盘索引，不重新导入。OV/embedding沿用19443/19444，RAG新19446；先检查端口/PID归属。密钥经既有provider凭据读取接口传stdin，绝不打印或推送，不修改Pi配置。服务命名/token/output独占不覆盖。新wheel在独立环境装验；如复用已验依赖，注明范围，不能称从零联网安装。

## 顺序、冻结与验收

1. 主agent先同步两PLAN/README入口到GitHub。子任务timing30和grounded_answer可在互斥文件并行；子任务只写指定实现及独立drafts双版，不写主PLAN/Git/服务。
2. 定向失败测试→实现→受影响回归，主agent集成API、budget和UI；完整tests回归及两角度独立review，修确定缺陷。
3. 锁subset manifest；复用服务/索引，必要健康和身份核对。开发6仅一组grounded candidate；不得使用正式30结果调参。仅已确认代码bug修复允许另run_id复核开发，不反复挑最好一次。
4. 在同一最终源码下预冻结baseline(text+threshold0)与candidate(grounded_json+grounded_evidence)所有配置、prompt、scope、model、manifest。两者相同检索/预算/生成主参数，差异是联合grounding策略，不能把变化全归因某个单因素。串行实际HTTP/SSE各跑固定30一次，全失败保留。
5. 分别核对raw/journal/report、gold、source/runtime、model/backend身份，按同query配对质量/引用/完成/等待/已知用量。TREC复核排名。摘录结构真实性和标签质量分开，不用语义支持率冒充人工评分。小样本不证明稳定前沿。
6. UI在正式之后做提问→可交付答案→原文→停止撤回；构建/独立安装/入口hash、实际服务身份分别留证。公开全部30×2及development6数字，raw/答案/论文/凭据/权重/私人路径留本机；秘密模式和内部文件不推。
7. 更新主双版、RESULTS/选型/简历安全措辞，独立review确认；扫描脱敏→普通commit/push→读取远端SHA与文件确认；一致DB备份、仅停核对属于本轮PID服务，不碰原Pi。

软件默认text保兼容。是否推荐新operator配置以小样本表现解释，质量没有改善不称改善、不保证达到预定分数。对照本身不能作为所有领域通用事实准确率或稳定泛化证明。

## 回退与待确认

保留b62887d及旧300所有材料。新模式operator可切回text+reranker_threshold；schema不改旧索引/DB，必要时用旧源码与保存Catalog备份。冻结报告不覆盖、不盲重放未知外部写入。生产目标、专家语义标签和跨领域规模属于后续，不是本轮必需输入。无待用户再次授权项。当前完整回归regression-v4为310 passed、5 warnings；两角度独立代码复审已修确定缺陷，正式30尚未开始。真实OpenViking/BGE/Qwen/产品服务已ready；开发6-v1/v2为评测修复前检查并完整保留，v3仅在评分/失败元数据bug修复后同策略复核。模型/索引/提示参数未调优。打包完成，安装检查进行中，网页/正式数据/最终资料同步待收尾。

## 实现补充与独立复审修复

- benchmarks/short_protocol.py在正式runner和audit前置核验完整development6报告/journal、两个selection互指与共同摘要、正式输出路径、source/config/manifest、运行模式和联合策略外配置一致性，不能只提供两个hash绕过；禁止重放已有正式输出。
- 科学benchmark的grounded模式要求exactly1验证事件，decision为三科学标签且与正文首行一致。一般问答ANSWER仍可交付，但在科学评分中记INVALID；流程完成与评分正确分开。每题不论原文/会话回读成功与否均保留验证元数据和已知usage。
- tools/audit_flow_timing.py独立从UTC/ns原始快照重建wall反跳/缺失与逐事件有效性，再复算stream/user/first与分位数和异常计数。时钟异常不删除质量样本；无时间时为未知而非零。
- tools/compare_short_user_flow.py同30配对，失败留分母，报告Wilson区间、paired bootstrap与exact McNemar；区分有摘录/无摘录的grounded响应及摘录数量。联合格式/门槛策略的收益不能归因单因素，串行缓存/网络条件不能支持速度因果结论。
