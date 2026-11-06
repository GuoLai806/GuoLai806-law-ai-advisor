# 如何复现真实 OpenViking 评测，以及如何复算公开数字

本文是 [执行版](PLAN-FOR-AI.md) 的操作补充，与 [用户版](PLAN-FOR-YOU.md) 使用同一组决定。分两件事：复算已经公开的数字不需要调用模型；在自己的机器重新跑完整流程，需要重新下载官方数据和固定模型、启动真实 OpenViking，并使用自己的云凭据。后者是新的运行记录，不能覆盖或代替本轮原始成绩。

下列命令按仓库现有 CLI 与源码参数编写，没有提供作者的私人配置。路径、端口占用和凭据由执行者填写，这是部署前置条件，不是新的业务选择。示例中的绝对路径均为占位符。不要把凭据放进命令参数、shell 历史、公共配置或日志。

## 1. 固定代码与工作目录

本轮生产代码及 benchmark 冻结提交为：

`7231a1fc00a20f5b311e083e7fcd2d5ed266dddb`

交付提交还包含 `tools/`、文档与示例。`benchmarks.e2e.source_hashes()` 覆盖 `evidence_rag/*.py`、`benchmarks/*.py`、`pyproject.toml` 和 `evidence_rag/static/index.html`。不能向这两个顶层 Python 目录增加辅助文件，也不能为跑通新环境而修改冻结代码。交付版的 `tools/`、`examples/`、`constraints-models-tested.txt` 可以复制进冻结 checkout，它们不属于该集合，但仍须单独记录 hash。原数据、模型缓存、配置和结果放在仓库外。

先填写三个独立的绝对路径；复现应使用空目录，不复用正在运行的本轮目录或服务：

```sh
export RAG_RUN="/absolute/path/to/your-private-rag-run"
export RAG_PUBLIC="/absolute/path/to/law-ai-advisor-delivery"
export RAG_CODE="/absolute/path/to/law-ai-advisor-frozen"
mkdir -p "$RAG_RUN"
git clone --branch feat/versioned-legal-rag-20261105 https://github.com/GuoLai806/GuoLai806-law-ai-advisor.git "$RAG_PUBLIC"
git -C "$RAG_PUBLIC" worktree add --detach "$RAG_CODE" 7231a1fc00a20f5b311e083e7fcd2d5ed266dddb
cp -R "$RAG_PUBLIC/tools" "$RAG_CODE/"
cp -R "$RAG_PUBLIC/examples" "$RAG_CODE/"
cp "$RAG_PUBLIC/constraints-models-tested.txt" "$RAG_CODE/"
cd "$RAG_CODE"
```

仓库中的交付记录给出最终远端提交。下载时还应记录自己的 `git rev-parse HEAD`，以及交付工具的 hash；不要只记分支名。若交付材料尚未推送完整，先等待材料完整，不临时创建替代配置。

## 2. 环境与两种复现的界限

本轮环境为 macOS arm64、M5 Pro、48 GiB、Python 3.12.13、Torch 2.15.0、Transformers 5.22.1、Sentence Transformers 6.2.0；重排与 embedding 使用 MPS/float16。共享桌面负载和云接口均影响耗时。更换硬件、设备、dtype 或依赖，必须另记环境，不能把新耗时当本轮耗时的复现。

```sh
python3.12 -m venv "$RAG_RUN/rag-env"
export RAG_PY="$RAG_RUN/rag-env/bin/python"
"$RAG_PY" -m pip install -c constraints-models-tested.txt ".[models,dev]"
python3.12 -m venv "$RAG_RUN/ov-env"
"$RAG_RUN/ov-env/bin/python" -m pip install openviking==0.5.3
export HF_HOME="$RAG_RUN/hf-cache"
export RAG_EVIDENCE="$RAG_PUBLIC/docs/development/openviking-user-flow-20261106/evidence"
mkdir -p "$RAG_RUN/results"
```

`constraints-models-tested.txt` 是经过测试的版本快照，不是跨平台、带 wheel hash 的完整锁文件；OpenViking 也要记录实际安装的全部依赖。保存两个环境的 `pip freeze`、OS/CPU/RAM、设备与 dtype；前后不升级环境。只复算公开数字时不需要启动模型，安装基础项目、NumPy 和分析工具依赖即可。

## 3. 下载官方数据并核对字节

BEIR 语料、查询与 qrels 使用公开下载入口，下载到仓库外。冻结 manifest 已存在，不覆盖它；下载命令生成另一份 manifest，随后核对关键字段：

```sh
"$RAG_PY" -m benchmarks.prepare --dataset scifact --data-dir "$RAG_RUN/beir-data" --manifest "$RAG_RUN/beir-download.manifest.json" --validation-count 32
```

数据应为 5,183 篇语料、300 道 BEIR test 查询，验证集为 train 中按 SHA256(query_id) 排序固定选择的 32 题。下载器核对官方 MD5，并记录 archive/corpus/queries/qrels 的 SHA256。执行时使用仓库中的 `benchmarks/manifests/scifact-validation.json`；将下载 manifest 的数据 hash、数量、验证 ID 与其核对，不修改冻结 manifest 来适应不同数据。

标签来自 [SciFact 官方数据说明](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/doc/data.md) 所指的官方压缩包。官方 URL 含 `latest`，因此必须核对字节，而不能只认 URL：

```sh
"$RAG_PY" - <<'PY'
import hashlib, os, tarfile, urllib.request
from pathlib import Path
root = Path(os.environ['RAG_RUN'])
archive = root / 'scifact-official.tar.gz'
expected = '11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'
payload = urllib.request.urlopen('https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz', timeout=60).read()
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Official archive bytes differ; preserve download and do not run the frozen evaluation')
with archive.open('xb') as stream:
    stream.write(payload)
destination = root / 'scifact-official'
destination.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
    stream.extractall(destination, filter='data')
PY
```

压缩包展开后使用 `data/claims_train.jsonl` 作 validation、`data/claims_dev.jsonl` 作本轮 test 标签。其 SHA256 应分别为 `f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0`、`86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217`。使用前核对每个被测 claim 的 ID 与文本精确对应 BEIR 查询；runner 会再次校验。

这是 SciFact 公开 dev 的标签，不是官方隐藏 test。gold 标签、证据文档 ID、`cited_doc_ids` 不得加入模型输入，不手动修正标注，不删除失败题。

## 4. 固定模型与生成 tokenizer

本轮公开模型的固定 revision 为：

| 用途 | 模型 | revision |
| --- | --- | --- |
| OpenViking embedding | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| 正式重排 | Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` |
| 最终验证对照 | BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

按 [公开模型示例](../../../examples/README.md) 下载固定 revision。BGE-M3 包含 `pytorch_model.bin`，Qwen 包含模板和模块 JSON，不能只下载部分权重后用其他版本 tokenizer 补齐：

```sh
"$RAG_PY" - <<'PY'
from huggingface_hub import snapshot_download
models = {
    'BAAI/bge-m3': '5617a9f61b028005a4858fdac845db406aefb181',
    'Qwen/Qwen3-Reranker-4B': '22e683669bc0f0bd69640a1354a6d0aebcfeede5',
    'BAAI/bge-reranker-v2-m3': '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
}
for model, revision in models.items():
    snapshot_download(model, revision=revision, allow_patterns=[
        '*.json', '*.safetensors', '*.bin', '*.txt', '*.model', '*.jinja', 'README.md', 'LICENSE*'
    ])
PY
```

生成 tokenizer 来自 [DeepSeek 官方 token 用量说明](https://api-docs.deepseek.com/quick_start/token_usage/) 链接的 `https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip`。本轮重新核对该官方下载与原压缩包字节相同：1,911,504 bytes，SHA256 `e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495`。配置中的 `tokenizer_revision` 是这个身份字符串，不是 Hugging Face commit。

```sh
"$RAG_PY" - <<'PY'
import hashlib, io, os, urllib.request, zipfile
from pathlib import Path
expected = {
    'tokenizer.json': '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614',
    'tokenizer_config.json': '841f8cf146e3f0ad1082594a31f68ecf7608c20467ef355081333d85bbaeb1cb',
}
payload = urllib.request.urlopen('https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip', timeout=60).read()
if hashlib.sha256(payload).hexdigest() != 'e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495':
    raise RuntimeError('Tokenizer archive identity changed')
destination = Path(os.environ['RAG_RUN']) / 'tokenizer'
destination.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for name, digest in expected.items():
        matches = [entry for entry in archive.namelist() if Path(entry).name == name]
        if len(matches) != 1:
            raise RuntimeError('Unexpected tokenizer archive layout')
        content = archive.read(matches[0])
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError('Tokenizer file identity changed')
        with (destination / name).open('xb') as stream:
            stream.write(content)
PY
```

保存实际缓存文件名、大小与 SHA256，核对公开的模型文件身份，然后在正式运行前后再比较。不能只记模型名字或 revision。云生成使用 `deepseek-v4-pro` 别名，记录实际返回 model/fingerprint/日期；固定本地 tokenizer 不代表知道云端全部模板、隐藏协议或固定了云权重，无法承诺答案逐字重现。

## 5. 只复算公开数字：到这里即可

此模式不启动任何服务，也不重新问 300 题。使用公开脱敏 report 与官方 qrels，重算标签、排名、失败数、等待分位和已报告用量。独立检索复核还需要 NIST trec_eval v9.0.8：

```sh
git clone --branch v9.0.8 https://github.com/usnistgov/trec_eval.git "$RAG_RUN/trec-eval"
make -C "$RAG_RUN/trec-eval"
"$RAG_PY" tools/analyze_user_flow.py --report "$RAG_EVIDENCE/openviking-formal-test-v1.json" --qrels "$RAG_RUN/beir-data/scifact/qrels/test.tsv" --trec "$RAG_RUN/trec-eval/trec_eval" --output "$RAG_RUN/results/public-formal-recalculated.analysis.json"
```

最终 validation 两组可分别使用 `qwen-final-validation-v2.json`、`bge-final-validation-v2.json` 和 `qrels/train.tsv` 复算。配对比较命令为：

```sh
"$RAG_PY" tools/compare_user_flow_validation.py --left "$RAG_EVIDENCE/qwen-final-validation-v2.json" --right "$RAG_EVIDENCE/bge-final-validation-v2.json" --output "$RAG_RUN/results/public-validation-paired.json"
```

输出中的输入 hash 会是脱敏副本 hash，与原报告 hash 不同。对比时核对数值、样本数、失败和统计规则，不要求输入 hash 与原报告相同。`PUBLIC-REDACTION-AUDIT.json` 登记原始与公开副本 hash，脱敏没有删题或改变数值。

公开副本没有原始 SSE、答案和论文正文，**不能独立重做“答案确实来自原始 SSE”或“原文确实与当时证据一致”的审计**。公开 audit 是原作者对本机原始材料的检查结果。数字能复算，不等于外部已重跑产品或完成语义引用核验。

## 6. 重新跑产品：启动真实 embedding 和 OpenViking

从这一节开始会运行模型并在第 9 节调用云 API。三个服务只监听 loopback，端口分别是 embedding 19444、OpenViking 19443、RAG 19445。执行者先确认端口未被其他任务占用；不要停止或覆盖别人的服务。使用三个独立终端，保持同一 `RAG_RUN`、`RAG_CODE`、`RAG_PY`、`HF_HOME` 路径。

第一个终端启动公开 BGE 服务；其源文件 SHA256 与本轮实际 endpoint 相同，为 `e8a05f535661ddfee34a2ec9204d0852f59f1f0dc2dbe033c1ff1b046d19e2ea`：

```sh
HF_HUB_OFFLINE=1 EMBEDDING_DEVICE=mps "$RAG_PY" -m uvicorn --app-dir examples bge_embedding_server:app --host 127.0.0.1 --port 19444
```

另一个终端将 `examples/openviking.local.json` 复制到仓库外。将 `storage.workspace`、`storage.parse_output.local_root`、`parser.download_temp_dir`、`ingest.state_dir` 改为本次私有目录，`embedding.dense.api_base` 改成 `http://127.0.0.1:19444/v1`，`server.port` 改为 19443。保留 1024 维、`text_source=content_only`、不使用 VLM 和遥测。不要把路径占位符直接交给 OpenViking 解析。配置自身记录 hash；不同本机路径会改变配置字节，因此记录与本轮配置的关系，不要求新配置 SHA 与作者本机配置相同。

```sh
NO_PROXY=127.0.0.1,localhost,::1 no_proxy=127.0.0.1,localhost,::1 "$RAG_RUN/ov-env/bin/openviking-server" --config "$RAG_RUN/openviking.local.json" --host 127.0.0.1 --port 19443
```

示例将 OpenViking 的输入估算裁剪阈值提高，完整输入由 BGE 端点按真实 tokenizer 拒绝超过 8,192 tokens 的输入。这不扩大模型窗口，也不提供无限输入或生产资源控制。本轮科学摘要不是复杂 PDF/OCR 或任意长文能力的验收。

## 7. 实际导入、回读、可见性检查与激活

使用一个全新的 SQLite catalog、OpenViking workspace 和输出名称。索引命令负责 staging、实际上传、reconcile 与落 journal；validate 完整通过后才激活 revision：

```sh
"$RAG_PY" -m benchmarks.index_openviking index --corpus "$RAG_RUN/beir-data/scifact/corpus.jsonl" --manifest benchmarks/manifests/scifact-validation.json --db "$RAG_RUN/catalog.sqlite3" --revision scifact_full_v3 --corpus-id scifact --url http://127.0.0.1:19443 --output "$RAG_RUN/results/index-v1.json" --concurrency 4
"$RAG_PY" -m benchmarks.index_openviking validate --corpus "$RAG_RUN/beir-data/scifact/corpus.jsonl" --manifest benchmarks/manifests/scifact-validation.json --db "$RAG_RUN/catalog.sqlite3" --revision scifact_full_v3 --corpus-id scifact --url http://127.0.0.1:19443 --output "$RAG_RUN/results/validate-v1.json" --concurrency 4
```

这两个命令会实际使用 OpenViking，不是预先算好的余弦矩阵。写入合同为 `processing_mode=vectors_only`、`args.parse_mode=no_split`；每个完整单元的 L2 地址是 `scope/unit_id/unit_id.md`，scope 为 `viking://resources/rag/scifact/scifact_full_v3/`。检查索引与验证报告的 complete、source_unchanged、5,183 单元数量、每个回读 hash 和可见性结果。

若结果未知或失败，先保留报告和 journal，按 `evidence_rag.viking_index reconcile` 查询真实状态，不能直接重发未知写入。新一次尝试使用新报告名，不能通过覆盖旧失败来制造一次成功导入。正式 benchmark 期间不重建索引，不切 revision。

## 8. RAG 使用公开冻结配置，私有项通过 stdin 补入

公开 `qwen-final-validation-v2.config.json` 的原字节 SHA256 为 `68ec5ae8cf77893d5506f807ec201d7adf425e8f2ef8334780503cb8f4a9cfc2`。`openviking-formal-test-v1.config.json` 是同一字节配置。保留原字节，不能用重新序列化的 JSON 冒充相同 hash。先在冻结 checkout 比较 `source_hashes()` 与其中的 `source_sha256`，并检查配置 SHA；不一致就停止。

```sh
export RAG_CONFIG="$RAG_EVIDENCE/qwen-final-validation-v2.config.json"
"$RAG_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
from benchmarks.e2e import source_hashes
path = Path(os.environ['RAG_CONFIG'])
content = path.read_bytes()
if hashlib.sha256(content).hexdigest() != '68ec5ae8cf77893d5506f807ec201d7adf425e8f2ef8334780503cb8f4a9cfc2':
    raise RuntimeError('Selected configuration bytes differ')
if json.loads(content)['source_sha256'] != source_hashes():
    raise RuntimeError('Frozen benchmark source differs')
PY
```

第三个终端运行以下启动方式。它从公开配置读取所有被测参数，再仅补本机 DB、tokenizer、OpenViking URL、模型基础 URL、云 key 和访问身份。服务的 `auth` 是 `{访问token:身份名称}`，不是反过来。`getpass` 要求有控制终端；不能在不能隐藏输入的环境中回退打印凭据。云 key 与本地访问 token 分开：

```sh
HF_HUB_OFFLINE=1 "$RAG_PY" - <<'PY'
import getpass, json, os, subprocess, sys
from pathlib import Path
with open('/dev/tty', 'r+'):
    pass
root = Path(os.environ['RAG_RUN'])
configuration = json.loads(Path(os.environ['RAG_CONFIG']).read_text())['service_configuration'].copy()
cloud_key = getpass.getpass('自己的 DeepSeek API key：')
access_token = getpass.getpass('本次本地 RAG 访问 token（runner 使用相同值）：')
if not cloud_key or not access_token:
    raise RuntimeError('Explicit credentials required')
configuration.update({
    'db': str(root / 'catalog.sqlite3'),
    'tokenizer': str(root / 'tokenizer'),
    'ov_url': 'http://127.0.0.1:19443',
    'model_url': 'https://api.deepseek.com',
    'model_key': cloud_key,
    'auth': {access_token: 'benchmark-owner'},
})
subprocess.run([sys.executable, '-m', 'benchmarks.serve_user_flow', '--port', '19445'],
               input=json.dumps(configuration), text=True, check=True)
PY
```

`model_url` 是服务基础 URL，客户端追加 `/chat/completions`，不填写包含凭据的 URL。执行者使用自己的官方可用 endpoint；若使用代理、不同 provider 或模型别名，另记身份，不能声称同模型复现。上例不将 key/token 写入文件或 argv；不要加调试打印 configuration，不把环境或进程内存转储发布。

真实服务需通过 `/health/ready`，并从带本地访问 token 的 `GET /api/v2/models` 核对全部 `runtime_identity`。runner 会逐字段校验。配置仍为 Qwen batch8/MPS/float16、Pro enabled/low、4条证据/6,000证据token、窗口32,768、framing512、输出8,192、主请求60秒，完全空响应才同证据 disabled 重试一次45秒/2,048。门槛0是本轮冻结行为，不是完成了高精度标定，不能为减少正式失败而中途降低它。

## 9. 先验证、冻结本机选择，再串行跑正式 300 题一次

有两条可接受的路径，执行前选择其一并写入新的运行协议：

- **复现本轮已选配置：** 只重新运行 Qwen validation32 检查流程，随后为本机生成一个记录原始选择、原始配置 hash 与新 validation 报告 hash 的 selection。不得根据这 32 题或正式题改配置；这是“复现已选 Qwen”，不是重新做 Qwen/BGE 选型。
- **完整重做选型过程：** 分别启动 Qwen 与 BGE，使用各自公开 final-v2 配置跑完整 validation32；只有同源码、同生成配置且每题有记录，才能用 `tools/select_user_flow.py` 作一次选择。如果新的验证选择了 BGE，必须如实记录新选择，不能把它说成复现原本选出的 Qwen。

以下 runner 示例仅输入本地访问 token；不会把 key/token 放进命令行。先设置 validation，Qwen 与 BGE分别使用不同结果名和自己的 `RAG_CONFIG`。BGE service需先停止属于本次验证的Qwen service，再在同端口按同一启动方式改用BGE公开配置，不碰其他服务：

```sh
export RAG_SPLIT=validation
export RAG_RESULT_TAG=reproduce-qwen-validation-v1
"$RAG_PY" - <<'PY'
import getpass, json, os, subprocess, sys
from pathlib import Path
with open('/dev/tty', 'r+'):
    pass
root = Path(os.environ['RAG_RUN'])
split = os.environ['RAG_SPLIT']
tag = os.environ['RAG_RESULT_TAG']
claims = root / 'scifact-official' / 'data' / ('claims_train.jsonl' if split == 'validation' else 'claims_dev.jsonl')
command = [sys.executable, '-m', 'benchmarks.run_user_flow',
    '--url', 'http://127.0.0.1:19445', '--data-dir', str(root / 'beir-data' / 'scifact'),
    '--manifest', 'benchmarks/manifests/scifact-validation.json', '--claims', str(claims),
    '--config', os.environ['RAG_CONFIG'], '--split', split,
    '--output', str(root / 'results' / (tag + '.json')), '--raw-dir', str(root / 'raw')]
if split == 'test':
    command.extend(['--selection', str(root / 'results' / 'selection-local.json')])
token = getpass.getpass('本次本地 RAG 访问 token：')
if not token:
    raise RuntimeError('Explicit token required')
completed = subprocess.run(command, input=json.dumps({'token': token}), text=True)
raise SystemExit(completed.returncode)
PY
```

validation结束后先审核本机原始材料，Qwen示例如下；BGE使用自己的报告、配置和输出名，不能把Qwen audit复制成BGE结果：

```sh
"$RAG_PY" tools/audit_user_flow.py --report "$RAG_RUN/results/reproduce-qwen-validation-v1.json" --config "$RAG_EVIDENCE/qwen-final-validation-v2.config.json" --manifest benchmarks/manifests/scifact-validation.json --claims "$RAG_RUN/scifact-official/data/claims_train.jsonl" --qrels "$RAG_RUN/beir-data/scifact/qrels/train.tsv" --raw-dir "$RAG_RUN/raw" --output "$RAG_RUN/results/reproduce-qwen-validation-v1.audit.json"
```

完整重做两组选择的实际命令如下；它要求两个完整32题报告、同源码、runtime未变、配置只在reranker身份上不同，且正式输出及journal尚不存在：

```sh
"$RAG_PY" tools/select_user_flow.py --reports "$RAG_RUN/results/reproduce-qwen-validation-v1.json" "$RAG_RUN/results/reproduce-bge-validation-v1.json" --configs "$RAG_EVIDENCE/qwen-final-validation-v2.config.json" "$RAG_EVIDENCE/bge-final-validation-v2.config.json" --output "$RAG_RUN/results/selection-local.json" --test-output "$RAG_RUN/results/reproduce-formal-test-v1.json"
```

原始 selection 中的作者本机绝对路径不能直接使用。公开 selection 也只用于说明原决定。复现已选Qwen的路径应在本机**新建**selection，保持选定配置原字节hash，附原始selection/report hash、自己的validation/report hash、选择记录创建时间、协议类型及本机绝对config路径；不要覆盖原始selection或把新记录宣称为原字节副本。runner实际核对的是selection中的config_sha256，完整选型要求应由选择工具和原始材料审计共同保障。

如果采用第一条路径，在新的Qwen validation32结束并完成其原始材料审计后，用下列代码新建本机选择记录；不与上面的完整两组选择命令同时执行：

```sh
"$RAG_PY" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
from benchmarks.e2e import file_hash, source_hashes
root = Path(os.environ['RAG_RUN'])
config_path = Path(os.environ['RAG_CONFIG']).resolve()
validation_path = root / 'results' / 'reproduce-qwen-validation-v1.json'
validation = json.loads(validation_path.read_text())
config = json.loads(config_path.read_text())
manifest = json.loads(Path('benchmarks/manifests/scifact-validation.json').read_text())
if any(path.exists() for path in [root / 'results' / 'selection-local.json',
        root / 'results' / 'reproduce-formal-test-v1.json',
        root / 'results' / 'reproduce-formal-test-v1.json.journal.jsonl']):
    raise RuntimeError('Selection must precede formal records; do not overwrite')
if config['service_configuration']['reranker'] != 'qwen' or file_hash(config_path) != '68ec5ae8cf77893d5506f807ec201d7adf425e8f2ef8334780503cb8f4a9cfc2':
    raise RuntimeError('This protocol only reproduces the original selected Qwen configuration')
if not (validation['split'] == 'validation'
        and validation['queries_expected'] == validation['queries_recorded'] == len(validation['rows']) == 32
        and [row['query_id'] for row in validation['rows']] == manifest['validation']['query_ids']
        and validation['source_unchanged'] and validation['runtime_identity_unchanged']
        and validation['source_sha256'] == config['source_sha256'] == source_hashes()
        and validation['config_sha256'] == file_hash(config_path)
        and all(validation['runtime_identity'].get(key) == value for key, value in config['runtime_identity'].items())):
    raise RuntimeError('Local validation/source/runtime identity mismatch')
selection = {
    'selected_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'protocol': 'reproduce_original_selected_qwen_without_retuning; not a new BGE/Qwen selection',
    'config_path': str(config_path), 'config_sha256': file_hash(config_path), 'chosen_reranker': 'qwen',
    'source_commit': '7231a1fc00a20f5b311e083e7fcd2d5ed266dddb', 'source_sha256': source_hashes(),
    'original_selection_sha256': '1db88912a0b618352eacaadbf497fff6188ef485b7f1ee507abdf8cc93bec9ef',
    'original_validation_report_sha256': 'ee1fb5bb37d1aedd3ba1181d2e6f289011a0ef6632e368e22f9814b5673b83dc',
    'local_validation_report_sha256': file_hash(validation_path),
}
with (root / 'results' / 'selection-local.json').open('x') as stream:
    stream.write(json.dumps(selection, indent=2) + '\n')
PY
```

正式启动前冻结模型文件、tokenizer、后端健康与配置身份，保存“before”记录；正式期间不跑浏览器演示、额外云问答、并发测试或其他模型工作。重新启动所选RAG服务，核对身份后，设置 `RAG_SPLIT=test`、`RAG_RESULT_TAG=reproduce-formal-test-v1`、`RAG_CONFIG` 为本机selection指向的原字节配置，执行同一runner代码一次。

runner每题串行走真实HTTP/SSE，随后回读证据原文与完成会话；保存原始SSE、逐题journal和最终报告。它不支持从中断journal继续跑。发生中断时保留原记录并明确标未完成；若另协议重跑，使用新的run名、保留旧结果，不挑最好的一次。非零退出码也可能只是题目失败；有全部300题报告时先检查失败数和source/runtime一致性，不把非零退出当理由自动重试全部题。

## 10. 原始结果分析、审计、网页验收与归档

全部300题结束后，保存“after”模型文件与后端身份，与“before”比较；不先重启服务再采身份。阶段对比只有来源前20完整时才可发布完整数值。分析命令与第5节相同，但report换成本机 `results/reproduce-formal-test-v1.json`。

完整原始证据审计使用现有工具：

```sh
"$RAG_PY" tools/audit_user_flow.py --report "$RAG_RUN/results/reproduce-formal-test-v1.json" --config "$RAG_CONFIG" --manifest benchmarks/manifests/scifact-validation.json --claims "$RAG_RUN/scifact-official/data/claims_dev.jsonl" --qrels "$RAG_RUN/beir-data/scifact/qrels/test.tsv" --raw-dir "$RAG_RUN/raw" --selection "$RAG_RUN/results/selection-local.json" --model-before "$RAG_RUN/results/model-before.json" --model-after "$RAG_RUN/results/model-after.json" --backend-before "$RAG_RUN/results/backend-before.json" --backend-after "$RAG_RUN/results/backend-after.json" --output "$RAG_RUN/results/reproduce-formal-test-v1.audit.json"
```

模型身份JSON结构为 `{"models": {模型名: {"revision":固定revision或null, "files": [{"name":相对文件名, "bytes":大小, "sha256":文件hash}]}}}`；模型名、文件排序和选取集合前后一致，generation-tokenizer记录两文件与revision=null。后端JSON记录embedding `/health`身份、endpoint源文件hash、私有OV配置hash、OV `/health`身份，采样字段前后一致，不把采样时间放进被逐字比较的身份。这些记录由本机执行者采集，不能复制作者的“before”记录来代替新机器实测。模型库存核对还要另比较公开的固定文件清单。

audit检查原始SSE hash、journal与最终report、首行判定、引用文档、每次用量、gold与source/runtime配置关系；passed表示材料一致，不表示答案语义正确。全部题和失败都留在分母中，未知用量不补0；已知用量价格参照不等于账单或总成本。

正式测量结束后，再用 `http://127.0.0.1:19445/` 做浏览器实操：输入本地访问token，提问、等真实答案完成、展开原文，再另测停止撤回草稿。截图与原始论文/答案内容留本机，记录完成或失败；这一步不混进300题延迟。它验证操作入口，不是多轮问答、复杂PDF或生产并发验收。

最后停止自己启动的三个任务服务，核对PID或在对应终端结束，不使用宽泛进程匹配杀服务。为SQLite用一致性backup方法归档，保存OVworkspace、模型身份、配置、原始SSE、journal、全部历史失败、最终报告、分析与审计。只发布经过脱敏的独立实现与数字证据；公开前扫描凭据、私人路径、答案/原论文正文，并登记原始和公开副本各自hash。

本轮正式300题已完整结束，分析、27项原始审计和浏览器验收均完成，结果见RESULTS.md。正式计时跨主机睡眠，逐题monotonic不是日历等待；本复现说明不承诺行业领先或云端逐字重现。重新运行必须同时记录日历时间并保证主机唤醒。

后续普通JSON接口修复并未重新跑300题。历史raw审计使用7231冻结checkout与交付版tools；在最终HEAD上source检查会正确指出与历史配置不同，不应绕过。180项回归已通过，最终安装验证如下。

## 最终构建与安装身份

最终交付源码提交c46cd3b5c2d8695c04b4c22ee4e8eaf8f096b151，正式冻结提交仍为7231a1f。wheel在v9构建，SHA256=04d7652444716a4a4ce4f7c4a44f193a9894b970e454b0813c276965b0b3f758；同字节wheel在v11独立环境安装，pip check通过，网页HTTP200、HTML/API hash与交付源码一致。v11复用v8已独立安装的依赖，逐文件核对2856个非缓存文件；不是全新联网依赖解析。v9下载超时/持续停滞和v10环境创建失败全部保留，不算RAG质量失败，不删除。入口probe未加载模型或重新跑300题。

[安装入口证据](evidence/wheel-clean-entry-v11.json)、[依赖来源与文件核对](evidence/installed-dependency-provenance-v11.json)、[代码和完整测试](evidence/delivery-validation-v1.json)分别保留。
