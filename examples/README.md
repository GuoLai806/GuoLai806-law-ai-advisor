# 本机可复现的模型与 OpenViking 入口

本例从本轮实际使用的BGE服务与OpenViking 0.5.3配置整理，去掉了用户路径和私有配置。它是loopback开发示例，没有生产身份/请求容量控制；不要直接开放到公网。示例不包含权重、凭据、原始评测语料。

先在独立Python3.12环境用 `pip install -c constraints-models-tested.txt ".[models]"` 安装项目的models可选依赖，并使用本轮冻结的模型库版本：torch2.15.0、transformers5.22.1、sentence-transformers6.2.0（完整环境记录另见freeze和测试报告）。把HF_HOME设在仓库外，下载固定revision：

```python
from huggingface_hub import snapshot_download
models = {
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
    "Qwen/Qwen3-Reranker-4B": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
}
for repo, revision in models.items():
    snapshot_download(repo, revision=revision, allow_patterns=[
        "*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.jinja",
        "README.md", "LICENSE*",
    ])
```

BGE-M3在该固定revision使用pytorch_model.bin；不能只下载safetensors。Qwen重排的chat_template.jinja和各模块JSON也必须保留。下载带宽/空间由模型实际文件决定，示例不会捆绑或重新发布它们。

从仓库根显式启动BGE端点，MPS须可用且不自动退回CPU；需要CPU演示时显式改device。模型仅从本地缓存加载：

```sh
HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps uvicorn --app-dir examples bge_embedding_server:app --host 127.0.0.1 --port 19344
```

GET /health给出固定model/revision/dimension/device/dtype；POST /v1/embeddings接受OpenAI兼容input，拒绝超过8,192模型tokens的整输入，不静默截断。每批最多64输入、内部batch8；归一化向量转float32传输。EVIDENCE_RAG_EMBEDDING_LOG可选指定私有证据文件，仅记输入hash/数量/耗时。基准中batch2、服务示例batch8是不同测量，不能直接比较延迟。

OpenViking建议独立环境 `pip install openviking==0.5.3`，将openviking.local.json复制到仓库外并把所有/absolute/private路径替换为自己独立目录，再启动：

```sh
NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 openviking-server --config /absolute/path/to/private/ov.json --host 127.0.0.1 --port 19343
```

配置采用content_only、vectors_only资源写入、1024维BGE端点，不调用VLM。索引与应用步骤见[执行版](../docs/development/general-evidence-rag-20261106/PLAN-FOR-AI.md)。本例与正式SciFact精确余弦参考后端分开，不把启动成功称为完整检索质量通过。

## 输入完整性与长文范围

OpenViking0.5.3的 [embedding配置](https://github.com/volcengine/OpenViking/blob/v0.5.3/openviking_cli/utils/config/embedding_config.py) 默认max_input_tokens为4096，表示估算裁剪阈值。本机安装源码的vectorize_file和embedder在provider前可裁剪正文并附标记；这不是段落分块。CPU审计确认SciFact全部5,183篇本轮序列化文本在默认估算下未改变，但它不证明完整OV索引或任意长文等价。

示例把该估算阈值显式提高到2147483647，以避免受控开发输入先被估算裁掉，再由BGE端点按实际tokenizer拒绝超过8,192 tokens的整输入。它不改变模型真实上下文能力，也不提供无限输入/资源保证；不得把这个配置直接当生产容量控制。当前通用dense单元是完整输入文档，长文应先准备为有来源定位的短JSONL单元；单段过长、复杂PDF版面/OCR与自动长文切块尚需专门实现和质量验收。API能存储PDF不等于所有长PDF已具备完整向量覆盖。

## 一份短语料的完整操作顺序

以下路径和身份由部署者提供，命令从仓库根运行；每次导入产生一个完整revision快照，不会自动向活动语料追加单个文件。添加文档时准备合并后的完整JSONL，再创建新revision。

```sh
python -m evidence_rag ingest --db /absolute/path/catalog.sqlite3 --input /absolute/path/corpus.jsonl --corpus-id demo --revision r1
```

查看这一revision的unit ID，不从问题集或qrels取ID：

```python
from evidence_rag.catalog import Catalog
cat = Catalog('/absolute/path/catalog.sqlite3')
print([row[0] for row in cat.db.execute('SELECT unit_id FROM units WHERE revision=? ORDER BY unit_id', ('r1',))])
cat.db.close()
```

对每个真实unit ID分别send后reconcile；如果失败，保留journal并检查，不重发未知结果。下面的UNIT_ID由上一步获得：

```sh
python -m evidence_rag.viking_index send --db /absolute/path/catalog.sqlite3 --revision r1 --unit-id UNIT_ID --url http://127.0.0.1:19343 --scope viking://resources/rag/demo/r1
python -m evidence_rag.viking_index reconcile --db /absolute/path/catalog.sqlite3 --revision r1 --unit-id UNIT_ID --url http://127.0.0.1:19343 --scope viking://resources/rag/demo/r1
python -m evidence_rag.viking_index validate --db /absolute/path/catalog.sqlite3 --revision r1 --url http://127.0.0.1:19343 --scope viking://resources/rag/demo/r1
python -m evidence_rag activate --db /absolute/path/catalog.sqlite3 --revision r1
```

这里是首次激活；有旧活动版本时activate必须提供 `--expected OLD_REVISION`。validate是全部映射/可见性检查，不是检索质量评测。应用配置中的OV_SCOPE也必须是同一corpus/revision；切换已有服务时先drain，不能只改数据库指针而保留旧scope。

启动API前由自己的受限配置设置EVIDENCE_RAG_DB、EVIDENCE_RAG_AUTH，以及OV_URL、OV_SCOPE、RERANKER=qwen等。缺少生成模型/tokenizer/阈值配置时可验证检索，严格ready仍不通过。普通检索请求是 `POST /api/v2/retrieve`，JSON至少含query与session_id，Bearer身份由服务端映射；不要在公共示例写真实token。模型SSE的完整预算/阈值配置见执行版。

本轮公开端点示例已真实启动并通过五项HTTP检查；选定Qwen重排+既有OV合成索引+云模型SSE也已通过。配置模板在替换为自有临时路径后通过OpenViking0.5.3验证；构造配置会创建目录，不能直接校验未替换的/absolute占位路径。首次测试夹具因此报只读文件系统错误，修正记录保留。

本轮验收后已停止任务专属测试服务，保留数据、journal和权重缓存。复现时按上述命令重新启动；本轮没有对生产服务或已安装pi做替换。
