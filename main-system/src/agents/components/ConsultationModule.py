#!/usr/bin/env python
"""
重构后的法律咨询组件 - 简化、稳定、易于维护
使用标准化数据接口，避免复杂的解析逻辑
"""
from typing import Optional, Dict, Any
import os
import time
import logging
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from ..models.DoubaoLLM import DoubaoLLM
from ..schema import ConsultationInput, ConsultationOutput, RetrievalResult

logger = logging.getLogger(__name__)


class ConsultationModule:
    """重构后的法律咨询组件 - 简单稳定"""

    def __init__(self):
        """初始化"""
        self.llm = DoubaoLLM()
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self._create_prompt(),
            verbose=True
        )

    def _create_prompt(self, with_context: bool = False) -> PromptTemplate:
        """创建prompt - 支持有上下文和无上下文两种模式"""
        if with_context:
            return PromptTemplate(
                template="""你是雅子，精通法律知识，擅长用温柔的语气回答用户的法律问题。

职责：
1. 基于提供的法律条文，对用户的问题进行准确分析
2. 给出清晰的法律建议
3. 指导用户如何保护自己的权益
4. 建议用户在复杂情况下咨询专业律师

回答格式要求：
- 分点说明
- 语言通俗易懂
- 避免使用专业术语过多
- 不要直接复制粘贴法律条文，而是用自己的话解释
- 保持温柔、耐心的语气

相关法律条文：
{legal_context}

当前对话历史：
{history}

问题：{input}

请用雅子的语气给出专业的法律建议。
""",
                input_variables=["history", "input", "legal_context"]
            )
        else:
            return PromptTemplate(
                template="""你是雅子，精通法律知识，擅长用温柔的语气回答用户的法律问题。

职责：
1. 对用户的问题进行准确分析
2. 给出清晰的法律建议
3. 指导用户如何保护自己的权益
4. 建议用户在复杂情况下咨询专业律师

回答格式要求：
- 分点说明
- 语言通俗易懂
- 避免使用专业术语过多
- 保持温柔、耐心的语气

当前对话历史：
{history}

问题：{input}

请用雅子的语气给出专业的法律建议。
""",
                input_variables=["history", "input"]
            )

    async def consult(
        self,
        query: str,
        retrieval_data: Any = None
    ) -> Dict[str, Any]:
        """
        咨询主入口 - 接受各种格式的输入，输出标准化结果

        Args:
            query: 用户问题
            retrieval_data: 检索到的法律条文（可以是dict、RetrievalResult或str）

        Returns:
            标准化的字典输出
        """
        start_time = time.time()

        try:
            # 1. 标准化输入数据
            input_data = ConsultationInput.create(query, retrieval_data)
            logger.info(f"处理咨询 - 查询: {query[:50]}... | 结果数: {input_data.retrieval_result.count if input_data.retrieval_result else 0}")

            # 2. 生成响应
            response = self._generate_response(input_data)

            # 3. 构建输出
            output = ConsultationOutput.success(
                response=response,
                references=input_data.retrieval_result.get_all_references() if input_data.retrieval_result else []
            )

            total_time = (time.time() - start_time) * 1000
            logger.info(f"咨询完成 - 耗时: {total_time:.1f}ms | 成功: {output.success}")

            return {
                "response": output.response,
                "references": output.references
            }

        except Exception as e:
            logger.error(f"咨询失败: {e}", exc_info=True)
            output = ConsultationOutput.failure(str(e))
            return {
                "response": output.response,
                "references": output.references
            }

    def _generate_response(self, input_data: ConsultationInput) -> str:
        """
        核心响应生成逻辑 - 简单、分层的策略
        """
        query = input_data.query
        retrieval_result = input_data.retrieval_result

        # 策略1: 有检索结果时，基于检索结果回复
        if retrieval_result and retrieval_result.is_valid:
            return self._response_from_retrieval(query, retrieval_result)

        # 策略2: 没有检索结果但有关键词，使用关键词匹配
        keyword_response = self._response_from_keywords(query)
        if keyword_response:
            return keyword_response

        # 策略3: 默认回复
        return self._default_response(query)

    def _response_from_retrieval(self, query: str, retrieval_result: RetrievalResult) -> str:
        """基于检索到的法律条文生成回复 - 使用LLM生成"""
        # 格式化检索到的法律条文作为上下文
        legal_context = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(retrieval_result.chunks)])

        try:
            # 使用带上下文的prompt调用LLM生成回答
            prompt = self._create_prompt(with_context=True)
            # 直接构造完整的prompt字符串
            prompt_str = prompt.format(
                legal_context=legal_context,
                history=self.memory.load_memory_variables({})["history"],
                input=query
            )
            response = self.llm.predict(prompt_str)
            return response
        except Exception as e:
            logger.error(f"LLM生成回复失败: {e}")
            #  fallback到硬编码回复
            return self._extract_summary(legal_context, query)

    def _extract_summary(self, ref_content: str, query: str) -> str:
        """根据问题类型生成对应的回答"""
        query_lower = query.lower()

        # 合同相关问题
        if "合同" in query_lower:
            if "违约" in query_lower:
                return "合同违约是指当事人不履行合同义务或履行不符合约定的情况。根据《民法典》规定，违约方需要承担继续履行、赔偿损失等责任。您可以先与对方协商，如协商不成可通过法律途径解决。"
            elif "纠纷" in query_lower:
                return "合同纠纷是指在合同订立、履行、变更、终止过程中产生的争议。常见类型包括违约责任、合同无效、解除合同等。处理合同纠纷的方式包括协商、调解、仲裁和诉讼。"
            else:
                return "合同是民事主体之间设立、变更、终止民事法律关系的协议，受法律保护。签订合同时应注意明确双方权利义务、违约责任等条款。"

        # 婚姻相关问题
        elif "离婚" in query_lower or "婚姻" in query_lower:
            return "婚姻关系受法律保护。离婚时夫妻共同财产由双方协议处理，协议不成时由法院根据财产情况、照顾子女和女方权益的原则判决。子女抚养问题也应考虑孩子的利益。"

        # 侵权相关问题
        elif "侵权" in query_lower or "赔偿" in query_lower:
            return "侵权责任是指行为人因过错侵害他人民事权益造成损害的，应当承担的法律责任。常见的侵权类型包括人身伤害、财产损失、知识产权侵权等。受害人有权要求侵权人停止侵害、赔偿损失等。"

        # 民法典基本原则问题
        elif "第一条" in query_lower or "第二条" in query_lower or "第三条" in query_lower:
            return "民法典是调整民事关系的基本法律，保护民事主体的合法权益，维护社会和经济秩序。它规定了平等、自愿、公平、诚信等基本原则，是我们处理民事事务的重要法律依据。"

        # 默认回答
        else:
            return f"关于您的问题「{query}」，我为您找到了相关的法律依据。您可以查看下方的引用条文来获取详细的法律规定。如果需要更详细的解读，建议咨询专业律师。"


    def _response_from_keywords(self, query: str) -> Optional[str]:
        """基于关键词的备用响应 - 直接使用硬编码回复以测试延迟"""
        logger.info("直接使用硬编码回复以测试延迟")
        return self._get_hardcoded_response(query)

    def _get_hardcoded_response(self, query: str) -> Optional[str]:
        """硬编码的备用响应"""
        query_lower = query.lower()

        # 民法典第一条相关
        if "第一条" in query or "立法目的" in query:
            return """您好！关于民法典第一条的内容，我来为您介绍一下：

1. **民法典第一条内容**：为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

2. **立法目的解读**：
   - 保护民事主体的合法权益
   - 调整民事关系
   - 维护社会和经济秩序
   - 适应中国特色社会主义发展要求
   - 弘扬社会主义核心价值观

3. **法律地位**：民法典是新中国成立以来第一部以"法典"命名的法律，被誉为"社会生活的百科全书"。

如果您还有其他问题，请随时告诉我！😊"""

        # 合同违约相关
        elif "违约" in query_lower or "合同" in query_lower:
            return """您好！关于合同违约的问题，我来为您解答一下：

1. **合同违约的定义**：根据《民法典》第五百七十七条的规定，当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。

2. **您可以采取的措施**：
   - 首先，可以与对方协商，要求继续履行合同或者赔偿损失
   - 如果协商不成，可以向法院提起诉讼或者向仲裁机构申请仲裁
   - 在诉讼或仲裁前，可以收集相关证据，如合同、通信记录、损失证明等

3. **诉讼时效**：一般来说，合同纠纷的诉讼时效是三年，从知道或者应当知道权利受到损害以及义务人之日起计算。

如果您需要更详细的法律建议，建议您咨询专业律师。希望我的回答能帮到您！😊"""

        # 离婚相关
        elif "离婚" in query_lower or "婚姻" in query_lower:
            return """您好！关于婚姻相关的法律问题，我来为您解答：

1. **离婚方式**：根据《民法典》规定，离婚有两种方式：
   - 协议离婚：夫妻双方自愿离婚的，应当签订书面离婚协议，并亲自到婚姻登记机关申请离婚登记
   - 诉讼离婚：夫妻一方要求离婚的，可以由有关组织进行调解或者直接向人民法院提起离婚诉讼

2. **财产分割原则**：
   - 离婚时，夫妻的共同财产由双方协议处理
   - 协议不成的，由人民法院根据财产的具体情况，按照照顾子女、女方和无过错方权益的原则判决

3. **子女抚养**：
   - 离婚后，不满两周岁的子女，以由母亲直接抚养为原则
   - 已满两周岁的子女，父母双方对抚养问题协议不成的，由人民法院根据双方的具体情况，按照最有利于未成年子女的原则判决

建议您根据具体情况咨询专业律师获取更详细的建议！😊"""

        # 侵权责任相关
        elif "侵权" in query_lower or "赔偿" in query_lower:
            return """您好！关于侵权责任的问题，我来为您解答：

1. **侵权责任的定义**：根据《民法典》第一千一百六十五条规定，行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。

2. **承担侵权责任的方式**：
   - 停止侵害
   - 排除妨碍
   - 消除危险
   - 返还财产
   - 恢复原状
   - 赔偿损失
   - 赔礼道歉
   - 消除影响、恢复名誉

3. **诉讼时效**：向人民法院请求保护民事权利的诉讼时效期间为三年。法律另有规定的，依照其规定。

如果您的权益受到侵害，建议及时咨询专业律师！😊"""

        # 民法典第二条/第三条相关
        elif "第二条" in query or "第三条" in query:
            return """您好！关于民法典的相关条文，我为您提供一些信息：

民法典的基本原则包括：
1. **平等原则**：民事主体在民事活动中的法律地位一律平等
2. **自愿原则**：民事主体从事民事活动，应当遵循自愿原则
3. **公平原则**：民事主体从事民事活动，应当遵循公平原则
4. **诚信原则**：民事主体从事民事活动，应当遵循诚信原则

如果您需要了解具体条文内容，建议您查阅《民法典》原文或咨询专业律师。😊"""

        return None

    def _default_response(self, query: str) -> str:
        """默认回复 - 直接使用硬编码以测试延迟"""
        logger.info("使用默认硬编码回复以测试延迟")
        return f"""您好！关于您的问题「{query}」，我来为您提供一些法律建议：

1. **问题分析**：您的问题涉及到法律相关领域。

2. **建议步骤**：
   - 首先，明确您的具体需求和问题焦点
   - 收集与问题相关的证据和材料
   - 可以先尝试与对方协商解决
   - 如协商不成，可以考虑通过法律途径解决

3. **温馨提示**：
   - 建议您咨询专业律师以获取更准确的法律意见
   - 注意保存相关证据和文件
   - 关注诉讼时效问题

如果您能提供更多细节，我可以为您提供更具体的建议！😊"""

    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear()

    def get_memory(self):
        """获取对话记忆"""
        return self.memory
