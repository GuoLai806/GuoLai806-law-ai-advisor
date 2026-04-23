from typing import Dict, Any


class DocumentGeneration:
    """法律文书生成组件"""

    async def generate(self, document_type: str, requirements: str) -> str:
        """根据用户要求生成法律文书"""
        if document_type not in document_templates:
            return f"不支持{document_type}类型的法律文书生成"

        # 查找相应的文档模板
        template = document_templates.get(document_type)
        if not template:
            return f"未找到{document_type}类型的文档模板"

        try:
            # 根据模板生成文档
            document = await template.generate(requirements)
            return self._format_document(document_type, document)

        except Exception as e:
            return f"生成{document_type}时出错: {e}"

    def _format_document(self, document_type: str, document: str) -> str:
        """格式化文档"""
        document_title = {
            "contract": "合同",
            "complaint": "起诉状",
            "legal_letter": "律师函"
        }.get(document_type, "法律文书")

        return f"""
=== {document_title} ===

{document}

---
注：此文档仅供参考，请咨询专业律师进行最终审核。
"""


# 文档模板定义
class DocumentTemplate:
    """文档模板基类"""
    async def generate(self, requirements: str) -> str:
        raise NotImplementedError


class ContractTemplate(DocumentTemplate):
    """合同模板"""
    async def generate(self, requirements: str) -> str:
        return f"""合同

根据您的需求，这是一个合同模板：

甲方：[甲方姓名/公司名称
地址：[甲方地址]

乙方：[乙方姓名/公司名称]
地址：[乙方地址]

根据《中华人民共和国民法典》及相关法律法规，甲乙双方本着平等、自愿、公平、诚信的原则，经友好协商，达成如下协议：

第一条 合同内容
{requirements}

第二条 合同期限
本合同期限为[合同期限]，自[开始日期]起至[结束日期]止。

第三条 权利与义务
[甲方权利与义务]

[乙方权利与义务]

第四条 违约责任
[违约责任条款]

第五条 争议解决
本合同履行过程中发生的争议，双方应友好协商解决；协商不成的，向有管辖权的人民法院起诉。

第六条 其他
1. 本合同自双方签字盖章之日起生效。
2. 本合同一式两份，甲乙双方各执一份，具有同等法律效力。

甲方（签字/盖章）：
日期：

乙方（签字/盖章）：
日期：
"""


class ComplaintTemplate(DocumentTemplate):
    """起诉状模板"""
    async def generate(self, requirements: str) -> str:
        return f"""民事起诉状

原告：[原告姓名]，[性别]，[出生日期]生，
住址：[原告住址]
联系电话：[原告电话]

被告：[被告姓名]，
住址：[被告住址]
联系电话：[被告电话]

诉讼请求：
1. [请填写具体诉讼请求1
2. [请填写具体诉讼请求2
3. 本案诉讼费用由被告承担。

事实与理由：
{requirements}

此致
[法院名称]人民法院

具状人：[原告姓名]
[日期]

附：本诉状副本[x]份
"""


class LegalLetterTemplate(DocumentTemplate):
    """律师函模板"""
    async def generate(self, requirements: str) -> str:
        return f"""律师函

律函字第[编号]号

致：[收件人姓名/单位]
地址：[收件人地址]

[律师事务所名称]接受[委托人姓名]的委托，指派[律师姓名]律师就贵方与[委托人姓名]之间的事宜，郑重致函如下：

一、事实概要
{requirements}

二、法律依据
[说明相关法律依据]

三、律师要求
1. [律师要求1
2. [律师要求2]

请贵方在收到本函后[期限天数]日内予以答复或履行上述义务，否则本律师将根据委托人的授权，通过法律途径追究贵方的法律责任。

特此函告！

[律师事务所名称]
律师：[律师姓名]
律师执业证号：[律师执业证号]
联系电话：[联系电话]
[日期]
"""


# 文档模板注册表
document_templates = {
    "contract": ContractTemplate(),
    "complaint": ComplaintTemplate(),
    "legal_letter": LegalLetterTemplate()
}
