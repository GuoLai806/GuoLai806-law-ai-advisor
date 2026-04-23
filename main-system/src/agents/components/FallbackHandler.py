class FallbackHandler:
    """未识别意图的处理组件"""

    def __init__(self):
        from ..models.DoubaoLLM import DoubaoLLM
        import os
        self.llm = DoubaoLLM(
            model_id=os.getenv("DOUBAO_MODEL_ID", "doubao-seed-2-0-mini-260215"),
            api_key=os.getenv("ARK_API_KEY", ""),
            base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
        )

    async def handle(self, input_message: str) -> str:
        """处理未识别意图的情况"""
        import time
        import logging
        logger = logging.getLogger(__name__)
        start_time = time.time()

        # 简单的关键词检测
        greetings = ["你好", "您好", "hi", "hello", "在吗", "在么"]
        if any(keyword in input_message.lower() for keyword in greetings):
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"⏱️ [Fallback] 问候检测耗时: {elapsed:.1f}ms")
            return "您好！我是雅子，您的法律小助手。有什么法律问题需要我帮忙吗？"

        logger.info(f"⏱️ [Fallback] 开始模拟回复，输入: {input_message[:20]}...")

        # 使用模拟回复替代真实API调用，避免超时
        time.sleep(0.05)  # 模拟50ms延迟

        # 针对常见法律问题的模拟回复
        mock_responses = {
            "什么是侵权责任": "侵权责任是指行为人因过错侵害他人民事权益造成损害的，应当承担的法律责任。根据《民法典》规定，侵权责任的承担方式包括停止侵害、排除妨碍、消除危险、返还财产、恢复原状、赔偿损失、赔礼道歉、消除影响、恢复名誉等。",
            "如何申请专利": "申请专利需要以下步骤：1) 确定专利类型（发明、实用新型、外观设计）；2) 进行专利检索；3) 准备申请文件；4) 提交申请；5) 缴纳申请费；6) 初步审查；7) 实质审查（仅发明专利）；8) 授权公告。建议您咨询专业专利代理机构办理。",
            "离婚财产如何分割": "离婚财产分割遵循以下原则：1) 男女平等原则；2) 照顾子女和女方利益原则；3) 有利生活，方便生活原则；4) 权利不得滥用原则；5) 夫妻一方所有的财产，在共同生活中消耗、毁损、灭失的，另一方不予补偿。具体分割会根据财产类型（婚前财产、婚后财产）和实际情况确定。"
        }

        # 匹配关键词返回对应回复
        for key, value in mock_responses.items():
            if key in input_message:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⏱️ [Fallback] 模拟回复耗时: {elapsed:.1f}ms")
                return value

        # 默认模拟回复
        default_response = "关于这个法律问题，建议您咨询专业律师以获取更详细的帮助。如果您有其他法律问题，我会尽力为您解答。"
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"⏱️ [Fallback] 模拟回复耗时: {elapsed:.1f}ms")
        return default_response
