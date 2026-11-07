# 本轮交付与运行状态

[用户说明](PLAN-FOR-YOU.md) · [执行依据](PLAN-FOR-AI.md) · [结果及限制](RESULTS.md)。

代码实现、439项v2回归、独立复审、wheel构建、仓库外安装、实际后端全流程及网页交互均已完成，准备好交给用户体验。用户本人是否接受尚待确认。普通GitHub同步为最后步骤，以远端提交与逐文件回读为准，不能用本地commit代替推送成功。

## 已验证范围

- C1共享连接/排空/取消隔离；C2独立预览、同版本原文、完整证据生成。
- 439 passed，6个机制检查是该总数子集；4个P2修复复审。测试不代表质量样本。
- wheel0.2.2及安装核心模块、HTML与源码hash一致；.pth明确复用模型依赖，未声称全新依赖解析环境。
- 实际OpenViking0.5.3/BGE-M3/Qwen及生成模型：6开发+同30题各两种路径，失败完整留存；结果各22/30正确、29/30流程，未证明前沿或整体提速。
- CUA安装网页第一开发题1134：先看资料→截短提示→打开完整原文→回答完成/引用核验；探索停止、回答检索期停止、改题取消旧预览、恢复均观察终态。未另实测云生成中途取消，未新增正式质量分母。
- 后端前后OpenViking健康、embedding身份（排除load_seconds诊断）、配置hash、catalog一致；API ready=true，missing=[]。[后端记录](evidence/backend-after.json)不证明未来可用性。
- SQLite online backup且integrity_check=ok；[备份摘要](evidence/backup-v1.json)。大DB/原文、模型权重、私有配置留本机。

[UI记录](evidence/ui-acceptance-v1.json)保存操作结果及本机截图/AX摘要；截图不上传公开仓库。停止回答时参考区旧查找提示尚未清理，已列非阻塞待办，主状态已停止、草稿撤回。

## 本机体验与恢复

正常入口 http://127.0.0.1:19447/ ，本机演示令牌 `loopback-demo`；只有loopback暴露，不是生产凭据或公网部署。服务使用已安装wheel，runner启动脚本沿用仓库benchmarks；生产API/retrieval/exploration/viking_index/HTML均从仓库外安装位置导入。当前体验是grounded_json + grounded_evidence，默认产品text未变。

登记进程：API3320/19447、OV1150/19443、embedding1106/19444。仅表示本次核验时PID；停止前必须重新核对命令、端口和归属，不能依旧PID盲杀。三个服务保留供用户体验；源服务1257/19446已停止。

本机任务目录 `work/dev/rag-exploration-20261107/` 保存install-venv、tested-source-v1.tar.gz、wheels、catalog-delivery-v1.sqlite3、backend-before/after、服务日志、完整raw和管理脚本；`outputs/rag-exploration-20261107/`保存三组report/config/audit/paired和ui。仓库内文件按[复现](REPRODUCTION.md)提供公开操作依据，远端开发不依赖本机绝对路径。重启按同脚本/安全stdin配置，不重复起已存在模型服务。

回退：OV_REUSE_CLIENT=0回逐次连接，EXPLORE_ENABLED=0关闭预览；仅停止核实属于本任务的进程后再改服务配置/重启。保留旧安装、原索引及一致DB备份，不直接覆盖活动DB，不恢复低版本schema到在用进程。无需合并PR、强推、发布、修改企业或其他项目。

## GitHub归档

代码、同一组双版文档、选型理由、复现、审阅及公开脱敏证据一并普通提交到feat/versioned-legal-rag-20261105。README提供入口。同步后必须远端SHA及全部改动文件字节回读；本机保存receipt。只有核实成功才在最终交付称已同步。

网页停止/改题所说“取消”指浏览器旧请求及界面状态失效；没有另观察服务端OpenViking任务是否立即终止，工程生命周期测试独立报告。
