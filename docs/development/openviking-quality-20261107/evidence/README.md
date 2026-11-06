# 公开证据索引

对应[结果报告](../RESULTS.md)、[用户说明](../PLAN-FOR-YOU.md)、[AI依据](../PLAN-FOR-AI.md)。配置原字节保留；5份报告及journal共78行，所有失败、gold、排序、时钟和已报用量保留，题目/答案/来源正文与本机路径脱敏。

- 正式：[baseline30](baseline30-v1.json)、[candidate30](candidate30-v1.json)、对应config/journal/selection/audit/analysis。
- 开发：[v1](candidate-dev6-v1.json)、[v2](candidate-dev6-v2.json)、[最终v3](candidate-dev6-v3.json)；v1/v2仅历史，v3用于正式冻结。
- 比较：[paired30](paired30-v1.json)、[引用/NEI区间](paired-additional-v1.json)、[附加复算脚本](recompute-paired-additional.py)。后者在本目录执行，仅输出JSON、不发模型请求、不覆盖材料，需要NumPy。
- 检索复算：[baseline analysis](baseline30-v1.analysis.json)、[candidate analysis](candidate30-v1.analysis.json)，TREC逐题舍入复核。
- 身份：tested-source、model-artifacts前后、backend前后及identity-audit、host-conditions。raw after额外load_seconds诊断，身份投影不变。
- 正常使用：[wheel导入](installed-identity-v3.json)、[安装API](installed-user-flow-v1.json)、[UI状态与原始截图hash](ui-validation-v1.json)，相关wheel/CLI日志；截图正文留本机。
- 工程回归：regression-v4.log/xml，310 passed；独立review结论见交付复审文档。
- [脱敏审计](PUBLIC-REDACTION-AUDIT.json)保存原报告/公开报告/journal/config摘要及数字失败字段一致性；ui与附加统计及日志属于导出后追加的公开辅助资料，最终公开文件清单另保存，不把它们冒称该工具原来19个aux。

原始报告hash与公开文件hash不同；selection的脱敏路径不能直接用于正式运行，其pair摘要指向原本机冻结文件。按[复现](../REPRODUCTION.md)重新准备公开数据和本机私有配置，重建pair。公开数字可复算，完整raw/语义答案审计需要本机原始证据，不能用脱敏数字替代人工支持性验收。
