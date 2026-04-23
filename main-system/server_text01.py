"""
雅子AI法律顾问 - 主服务文件
基于FastAPI的法律智能顾问服务

作者: GuoLai
版本: 1.0.0
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.agents.LegalAgent import LegalAgent
from src.agents.components.OptimizedRAGSystem import OptimizedRAGSystem
from src.agents.context.AgentContext import IntentType

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 初始化应用
app = FastAPI(
    title="雅子AI法律顾问",
    description="一个基于AI的智能法律咨询服务",
    version="1.0.0"
)

# 允许CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局实例
legal_agent = None
rag_system = None


@app.on_event("startup")
async def startup():
    """启动时初始化"""
    global legal_agent, rag_system

    logger.info("正在初始化雅子AI法律顾问服务...")

    try:
        # 初始化法律智能体
        legal_agent = LegalAgent()
        logger.info("LegalAgent 初始化成功")

        # 初始化RAG系统
        rag_system = OptimizedRAGSystem()
        rag_system.initialize()
        logger.info("RAG系统初始化成功")

        logger.info("服务启动完成！")

    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页"""
    try:
        # 读取静态文件目录下的index.html
        if os.path.exists("static/index.html"):
            with open("static/index.html", "r", encoding="utf-8") as f:
                return f.read()
        else:
            # 如果找不到文件，返回一个简单的提示
            return """
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>雅子 · AI法律顾问</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
                    .container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #1a1a1a; text-align: center; }
                    .error { background: #ffebee; color: #c62828; padding: 10px; border-radius: 4px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>雅子AI法律顾问</h1>
                    <div class="error">首页文件未找到。请检查static目录下是否存在index.html文件。</div>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"首页渲染失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def stream_chat(request: Dict[str, Any]):
    """流式聊天接口（模拟逐字输出）"""
    if not legal_agent:
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        message = request.get("message", "")
        model_type = request.get("model_type", "local")

        if not message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        logger.info(f"收到消息: {message[:50]}... (模型类型: {model_type})")

        # 处理用户输入
        await legal_agent.process_input(message)

        # 获取响应
        response = legal_agent.get_output()
        intent_type = legal_agent.get_intent_type()
        error = legal_agent.get_error()

        if error:
            logger.error(f"处理消息时出错: {error}")
            raise HTTPException(status_code=500, detail=error)

        logger.info(f"响应生成成功 (意图类型: {intent_type})")

        # 模拟流式输出
        async def generate():
            import asyncio
            chars = list(response)
            for i, char in enumerate(chars):
                yield f"data: {char}\n\n"
                # 随机暂停来模拟打字速度
                await asyncio.sleep(0.03)
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"聊天接口出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/api/chat")
async def api_chat(request: Dict[str, Any]):
    """API聊天接口（为前端设计，保留向后兼容）"""
    if not legal_agent:
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        message = request.get("message", "")
        model_type = request.get("model_type", "local")

        if not message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        logger.info(f"收到消息: {message[:50]}... (模型类型: {model_type})")

        # 处理用户输入
        await legal_agent.process_input(message)

        # 获取响应
        response = legal_agent.get_output()
        intent_type = legal_agent.get_intent_type()
        error = legal_agent.get_error()

        if error:
            logger.error(f"处理消息时出错: {error}")
            raise HTTPException(status_code=500, detail=error)

        logger.info(f"响应生成成功 (意图类型: {intent_type})")

        return {
            "response": response,
            "model_type": model_type,
            "intent_type": intent_type.value if intent_type else "unknown",
            "references": legal_agent.get_references() if hasattr(legal_agent, 'get_references') else []
        }

    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"聊天接口出错: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


@app.post("/chat")
async def chat(request: Dict[str, Any]):
    """聊天接口（保持向后兼容）"""
    return await api_chat(request)


@app.get("/health")
async def health():
    """健康检查"""
    global legal_agent, rag_system

    health_status = {
        "status": "healthy",
        "service": "雅子AI法律顾问",
        "version": "1.0.0"
    }

    # 检查LegalAgent
    if legal_agent:
        health_status["legal_agent"] = "ready"
    else:
        health_status["legal_agent"] = "not_ready"
        health_status["status"] = "unhealthy"

    # 检查RAG系统
    if rag_system:
        rag_stats = rag_system.get_statistics()
        health_status["rag_system"] = {
            "status": "ready",
            "document_count": rag_stats.get("document_count", 0),
            "persist_dir": rag_stats.get("persist_dir", "unknown")
        }
    else:
        health_status["rag_system"] = "not_ready"
        health_status["status"] = "unhealthy"

    return health_status


@app.get("/stats")
async def get_statistics():
    """获取系统统计信息"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG系统未初始化")

    try:
        stats = rag_system.get_statistics()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # 检查是否有民法典PDF
    if not os.path.exists("中华人民共和国民法典 - 中华人民共和国最高人民法院.pdf"):
        logger.warning("未找到民法典PDF文件，RAG系统将无法提供法律条文支持")

    # 启动服务
    uvicorn.run(
        "server_text01:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
