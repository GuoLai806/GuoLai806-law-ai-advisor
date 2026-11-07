# RAG C1/C2 开发执行依据

[用户版](PLAN-FOR-YOU.md)。历史研究保持原样，依据见 [研究](../bvio-superagent-research-20261107/PLAN-FOR-AI.md)。

## 基线及范围
仓库 GuoLai806/GuoLai806-law-ai-advisor；分支 feat/versioned-legal-rag-20261105；基线 55139899cea9112908f7eddc27ba015c1d6a39d9；开工时工作区干净，当前为本轮候选改动，提交状态由远端归档核实。仅 C1/C2，其他项目、RAG C3/C4 待办。

## 需求与验收映射
- R1 保留真实 OV + BM25 + RRF + Qwen、scope/revision/profile/serialization/validated：旧测试回归和真实身份/结果对照。
- R2 C1 共享 httpx.AsyncClient，可注入 client 与 ownership；aclose 排空，关闭后拒绝，取消隔离；transport 向后兼容，trust_env=False、redirect=False、逐请求 timeout/headers不丢。生命周期测试及同30题真实 find AB/BA，对照每次新建和复用，失败留分母。
- R3 C2 POST /api/v2/explore：独立 ExploreBundle，无 evidence/ready 生成入口；字段 unit_id/document_id/revision/content_sha256/document_sha256/locator/part_no/preview/preview_truncated/has_more/read_pointer，read_pointer 同 revision 的 /api/v2/units/{id}。完整正文参与重排，探索绕过生成预算后仅输出有限预览；生成继续走原完整证据与预算。
- R4 网页可选先看资料、打开原文、问答和停止；过期请求不得覆盖新状态，DOM textContent，原文版本/hash核验。
- R5 开发6/正式30，每组固定同grounded_json + grounded_evidence；数据版本与模型配置不随错题调整；保存原始SSE/查询/引用/失败/时间/usage/源码和运行身份。另6个机制场景独立统计，不冒充泛化。正式小样本不证明前沿。
- R6 构建wheel、隔离安装、真实服务及网页验收；旧安装保留、DB备份；代码/测试/构建/安装/真实使用分别记录。
- R7 两版同提交、README入口、公开脱敏证据、远端SHA和文件回读；企业源码/凭据/原始私密配置不上Git。

## 文件与接口
retrieval.py: OpenViking lifecycle/client；exploration.py: 独立预览类型与转换；api.py: lifespan配置和关闭、新路由、运行身份；static/index.html: 可选先看资料与固定版本回读；serve_user_flow.py: 白名单配置；benchmarks新增C1对照与C2用户路径；tests/v2独立测试。
共享文件/API/Git/模型/服务/索引仅主Agent操作；子Agent写指定独立文件。依赖沿用httpx、Pydantic和已有模型，无新增平台。

## 顺序、测试和交付
先C1实现/对照，再C2集成；分别归因。必要回归和独立评审后冻结候选；6题开发检查后正式每组30题。读取真实OV0.5.3、SciFact5183篇、BGE-M3/Qwen本机MPS及既有私有模型配置，先核实PID不重复启动。正式T2必须从探索/直接入口到模型SSE、引用同版本原文和会话回读，组件mock只证明契约。计时使用monotonic及UTC/ns，保留失败；原始记录仓库外，公开发布必要脱敏数字与hash。

## 部署、回退及授权
仅本机loopback隔离安装；普通提交推送已授权，不合并PR/force/生产发布/发消息/改企业库。新增shared-client默认启用，可用配置回退逐次连接；explore可关闭；默认text不改变。旧wheel/服务配置/DB备份和旧index保留，不改schema。所有新进程登记到任务，本轮单一操作者。

## 待办与未知
其他项目、C3/C4、公开长文大规模质量评测、中文泛化和简历最终改写待办。云模型alias权重不可固定，网络复用和探索性能收益待实测，不承诺前沿。无需用户再次授权普通实施。最终用户本人体验验收仍待确认。

## 实施结果与当前收尾状态
C1/C2及索引生命周期实现完成；tests/v2 439通过，独立评审4个P2修复复审通过。已构建/安装0.2.2，生产模块+HTML同源hash，.pth模型依赖明确复用。C1真实60请求/30pair成功且同结果，网络中位24.7→21.3ms。开发6流程全通过/分类4正确；正式direct与explore各30、分类均22、流程均29、检索指标30/30相同，两份原始audit均通过。引用文档标签F1 .6667→.7143不能归因为探索质量收益；探索全流程P50 53.55s、direct40.59s，因此保留opt-in，不宣称整体提速/前沿。探索1条usage不全，已报tokens不能证明成本更低。详情/证据/统计定义见[RESULTS](RESULTS.md)。安装网页真实交互已通过：首开发题预览/原文/生成，探索停止、回答检索期停止、改题取消旧探索及恢复。最终后端身份及catalog与before一致，SQLite online backup integrity_check=ok；证据见[交付](DELIVERY.md)。GitHub 已同步代码及本组文档/证据，提交 [8da92ed](https://github.com/GuoLai806/GuoLai806-law-ai-advisor/commit/8da92ed38da772f9d0203fa2e2970da38fb369ed)；62 个改动文件已从远端逐一回读，内容与本地提交一致。[回读证据](evidence/github-code-sync-v1.json)。用户本人验收仍待体验确认。

计时还有一项重要限制：直接组一题的日历等待约19.3分钟，单调时钟约2分钟，差约17.3分钟；原因未知，记录保留且没有扣除。其他分位数不能遮住这条最大等待，两组速度差不能作为稳定提速证明。

部署入口19447；真实OV19443、embedding19444仅loopback；演示令牌loopback-demo属于本机隔离测试，不用于生产。GUI用grounded模式；没有改产品text默认。UI停止回答只实测了检索期撤回，云生成中途取消另由既有行为测试覆盖，不冒称额外模型实测。GUI stopped时参考区旧提示为非阻塞待办。恢复按DELIVERY及REPRODUCTION，不覆盖旧安装/索引，不推大DB/权重/凭据。

网页停止/改题所说“取消”指浏览器旧请求及界面状态失效；没有另观察服务端OpenViking任务是否立即终止，工程生命周期测试独立报告。

网页该次完成答案已由安装API会话接口回读，query/answer/revision与catalog一致，见[evidence/ui-session-readback-v1.json](evidence/ui-session-readback-v1.json)。首次不限定时间查到了继承库旧turn，选择断言失败记录保留；以截图前60秒到后1秒限定本次UI完成turn后唯一匹配，不删旧记录。只读回读不计新质量题。
