from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Literal
import torch
import fitz
import dashscope
from http import HTTPStatus
import os
import time
from pathlib import Path
import os

# 从环境变量读取配置，避免硬编码
m_openai_api_key = os.getenv("OPENAI_API_KEY", "")
m_base_url = os.getenv("OPENAI_BASE_URL", "https://aa.qinkeapi.cn/v1")
m_model = os.getenv("OPENAI_MODEL", "gemini-3.0-flash")

# 本地模型配置
LOCAL_MODEL_PATH = "models/lingminai/GuoLai-Law-Model"

# RAG配置
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
CIVIL_CODE_PDF_PATH = "中华人民共和国民法典 - 中华人民共和国最高人民法院.pdf"
RAG_PERSIST_DIR = "./chroma_civil_code"

class LocalLawModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载本地模型到设备: {self.device}")
        
        try:
            # 加载分词器和模型
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_PATH,
                torch_dtype="auto",
                device_map="auto"
            )
            
            print("本地法律模型加载成功")
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            self.model = None
    
    def generate(self, user_input: str):
        if not self.model:
            return "本地模型暂不可用，请切换到线上模型"
        
        try:
            # 使用您提供的专业系统提示
            system_prompt = """你是一个专业的法律顾问助手，由GuoLai开发。

你的特点：
- 专业知识：熟悉中国法律体系，能够提供专业的法律咨询
- 回答风格：详细、准确、有条理，提供完整的法律分析
- 回答格式：使用清晰的段落结构，必要时分点说明
- 法律依据：引用相关法律条文，提供具体的法律建议

当用户咨询法律问题时，你应该：
1. 首先理解用户的问题和情境
2. 分析涉及的法律领域和可能的法律条文
3. 提供详细的法律分析和建议
4. 必要时提醒用户咨询专业律师

你由GuoLai开发，旨在提供专业的法律咨询服务。"""
            
            # 构建对话格式 - 使用标准的 <|im_start|> 格式
            messages = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            # 直接使用transformers pipeline生成
            inputs = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=2048)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取assistant的回复部分
            if "<|im_start|>assistant" in result:
                result = result.split("<|im_start|>assistant")[-1]
            
            # 去除可能的结束标记
            if "<|im_end|>" in result:
                result = result.split("<|im_end|>")[0]
            
            # 过滤掉**这种标记符号
            result = self.filter_markdown_symbols(result)
            
            return result.strip()
            
        except Exception as e:
            print(f"本地模型生成错误: {str(e)}")
            return f"本地模型生成失败: {str(e)}"
    
    def filter_markdown_symbols(self, text):
        """过滤掉Markdown标记符号"""
        import re
        
        # 过滤掉** **这种标记
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # 过滤掉* *这种标记
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # 过滤掉__ __这种标记
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # 过滤掉#标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 过滤掉`代码`标记
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # 过滤掉多余的换行和空格
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text

class BailianEmbeddings:
    """阿里云百炼嵌入类"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key
        
    def embed_documents(self, texts):
        """批量嵌入文档"""
        try:
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=texts
            )
            
            if resp.status_code == HTTPStatus.OK:
                return [item['embedding'] for item in resp.output['embeddings']]
            else:
                raise Exception(f"API调用失败: {resp}")
                
        except Exception as e:
            print(f"批量嵌入异常，降级为逐个嵌入: {e}")
            return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str):
        """单个查询嵌入"""
        try:
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4", 
                input=text
            )
            
            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise Exception(f"API调用失败: {resp}")
                
        except Exception as e:
            print(f"单个嵌入异常: {e}")
            return [0.0] * 1024

class CivilCodeRAGSystem:
    """民法典RAG系统"""
    
    def __init__(self):
        self.vector_store = None
        self.embeddings = BailianEmbeddings(BAILIAN_API_KEY)
        self.initialized = False
        
    def initialize(self):
        """初始化RAG系统"""
        if self.initialized:
            return True
            
        try:
            # 检查是否需要重新处理
            if not self._needs_reprocessing():
                print("加载现有向量数据库...")
                self.vector_store = Chroma(
                    persist_directory=RAG_PERSIST_DIR,
                    embedding_function=self.embeddings
                )
                print("向量数据库加载成功")
            else:
                print("需要重新处理PDF，创建向量数据库...")
                self._create_vector_store()
                
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"RAG系统初始化失败: {e}")
            return False
    
    def _needs_reprocessing(self):
        """检查是否需要重新处理"""
        if not os.path.exists(RAG_PERSIST_DIR):
            return True
            
        pdf_mtime = os.path.getmtime(CIVIL_CODE_PDF_PATH) if os.path.exists(CIVIL_CODE_PDF_PATH) else 0
        db_files = list(Path(RAG_PERSIST_DIR).rglob("*"))
        
        if not db_files:
            return True
            
        db_mtime = max(os.path.getmtime(f) for f in db_files if os.path.isfile(f))
        return pdf_mtime > db_mtime
    
    def _create_vector_store(self):
        """创建向量数据库"""
        # 提取PDF文本
        documents = self._extract_pdf_text()
        if not documents:
            raise Exception("PDF提取失败")
        
        # 文本分块
        chunks = self._chunk_text(documents)
        
        # 创建向量数据库
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=RAG_PERSIST_DIR
        )
        print(f"向量数据库创建成功，包含 {len(chunks)} 个文档")
    
    def _extract_pdf_text(self, max_pages: int = 10):
        """提取PDF文本内容"""
        documents = []
        
        try:
            doc = fitz.open(CIVIL_CODE_PDF_PATH)
            pages_to_process = min(max_pages, len(doc))
            
            for page_num in range(pages_to_process):
                text = doc[page_num].get_text()
                if text.strip():
                    from langchain.schema import Document
                    documents.append(Document(
                        page_content=text,
                        metadata={"page": page_num + 1}
                    ))
            
            doc.close()
            print(f"提取了 {len(documents)} 页PDF内容")
            return documents
            
        except Exception as e:
            print(f"PDF提取错误: {e}")
            return []
    
    def _chunk_text(self, documents):
        """文本分块处理"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        return text_splitter.split_documents(documents)
    
    def search_civil_code(self, question: str, k: int = 3):
        """搜索民法典相关内容"""
        if not self.initialized or not self.vector_store:
            return []
            
        try:
            results = self.vector_store.similarity_search(question, k=k)
            return results
        except Exception as e:
            print(f"RAG搜索失败: {e}")
            return []
    
    def format_rag_context(self, question: str) -> str:
        """格式化RAG上下文"""
        if not self.initialize():
            return ""
            
        results = self.search_civil_code(question, k=2)
        if not results:
            return ""
            
        context_parts = []
        for i, doc in enumerate(results, 1):
            # 清理文本，去除过多空格和换行
            content = ' '.join(doc.page_content.split())
            context_parts.append(f"相关法律条文 {i}: {content[:200]}...")
        
        return "\n".join(context_parts)

class ChatBot:
    def __init__(self, model_type: Literal["online", "local"] = "online"):
        self.model_type = model_type
        self.rag_system = CivilCodeRAGSystem()
        
        # 增强系统提示，包含RAG功能说明
        self.system_prompt = """
你是雅子，精通法律知识，擅长用温柔的语气回答用户的法律问题。

你现在可以访问民法典数据库，能够基于具体的法律条文为用户提供准确的法律咨询。

请用雅子的语气回复用户，并基于民法典条文提供专业建议。
"""
        
        if model_type == "online":
            self.llm = ChatOpenAI(
                model=m_model,
                temperature=0.9,
                openai_api_key=m_openai_api_key,
                base_url=m_base_url
            )
        else:
            self.local_model = LocalLawModel()
    
    def set_model_type(self, model_type: Literal["online", "local"]):
        self.model_type = model_type
        if model_type == "online" and not hasattr(self, 'llm'):
            self.llm = ChatOpenAI(
                model=m_model,
                temperature=0.9,
                openai_api_key=m_openai_api_key,
                base_url=m_base_url
            )

    def chat(self, user_input:str) -> str:
        try:
            # 获取RAG上下文（适用于所有模型）
            rag_context = self.rag_system.format_rag_context(user_input)
            
            # 构建增强的提示词
            enhanced_prompt = self.system_prompt
            if rag_context:
                enhanced_prompt += f"\n\n以下是相关的民法典条文:\n{rag_context}\n\n请基于以上条文回答用户问题。"
            
            if self.model_type == "online":
                messages = [
                    ("system", enhanced_prompt),
                    ("user", user_input)
                ]
                result = self.llm.invoke(messages)
                # 对线上模型输出也进行过滤
                return self.filter_markdown_symbols(result.content)
            else:
                # 确保本地模型已初始化
                if not hasattr(self, 'local_model') or self.local_model is None:
                    self.local_model = LocalLawModel()
                
                # 为本地模型构建增强的输入
                enhanced_input = user_input
                if rag_context:
                    enhanced_input = f"问题: {user_input}\n\n相关法律条文:\n{rag_context}\n\n请基于以上条文回答:"
                
                # 使用本地模型生成回答
                return self.local_model.generate(enhanced_input)
        except Exception as e:
            return f"哎呀，妈妈有点没听清楚呢: {str(e)}"
    
    def filter_markdown_symbols(self, text):
        """过滤掉Markdown标记符号"""
        import re
        
        # 过滤掉** **这种标记
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # 过滤掉* *这种标记
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # 过滤掉__ __这种标记
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # 过滤掉#标题标记
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # 过滤掉`代码`标记
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # 过滤掉多余的换行和空格
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text

app = FastAPI(title="雅子妈妈", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/")
async def root():
    return FileResponse("static/index_text01.html")

@app.get("/text01")
async def text01_page():
    return FileResponse("static/index_text01.html")

# 延迟初始化，避免导入时API密钥未设置的问题
chatbot_instance = None

def get_chatbot_instance():
    """获取ChatBot实例（延迟初始化）"""
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = ChatBot()
    return chatbot_instance

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_type: Literal["online", "local"] = "online"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_type: str

class ModelSwitchRequest(BaseModel):
    session_id: str
    model_type: Literal["online", "local"]

sessions = {}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(hash(request.message))
        if session_id not in sessions:
            sessions[session_id] = ChatBot(model_type=request.model_type)
        else:
            # 如果会话已存在，检查是否需要切换模型
            if sessions[session_id].model_type != request.model_type:
                sessions[session_id].set_model_type(request.model_type)
        
        bot = sessions[session_id]
        response = bot.chat(request.message)
        return ChatResponse(
            response=response, 
            session_id=session_id,
            model_type=bot.model_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/switch_model")
async def switch_model(request: ModelSwitchRequest):
    try:
        if request.session_id in sessions:
            sessions[request.session_id].set_model_type(request.model_type)
            return {"message": f"模型已切换到: {request.model_type}"}
        else:
            raise HTTPException(status_code=404, detail="会话不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

def main():
    import uvicorn
    import os
    
    # 从环境变量读取端口，默认8000
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
