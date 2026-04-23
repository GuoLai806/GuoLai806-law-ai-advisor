# 雅子AI法律顾问

基于RAG技术的智能法律咨询系统。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`

2. 配置环境变量：创建 `.env`（需要 BAILIAN_API_KEY 和 DOBAO_API_KEY）

3. 启动服务：`python server_text01.py`

4. 访问：http://localhost:8000

## 项目结构

```
law_adviser/
├── src/
│   └── agents/          # 核心逻辑
├── static/              # 前端页面
├── chroma_civil_code/    # 向量数据库
├── bm25_civil_code/    # BM25索引
├── server_text01.py      # 服务入口
└── requirements.txt      # 依赖
```
