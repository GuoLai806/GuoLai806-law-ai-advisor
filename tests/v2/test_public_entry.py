"""Public launch paths must use the shared evidence-first service."""
import json,subprocess,sys,tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def test_legacy_entry_imports_only_shared_legal_app():
    script="""import json,sys,server_text01
from legal_rag.api import app
print(json.dumps({'same_app':server_text01.app is app,'old_agent_loaded':'src.agents.LegalAgent' in sys.modules,'torch_loaded':'torch' in sys.modules}))"""
    p=subprocess.run([sys.executable,'-c',script],cwd=ROOT/'main-system',capture_output=True,text=True,check=True)
    data=json.loads(p.stdout.strip());assert data=={'same_app':True,'old_agent_loaded':False,'torch_loaded':False}

def test_entrypoints_and_runtime_dependencies_are_explicit():
    meta=tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert meta['project']['scripts']=={'evidence-rag':'evidence_rag.cli:main','legal-rag':'legal_rag.cli:main'}
    assert {'dev','models'} <= set(meta['project']['optional-dependencies'])
    assert not any('langchain' in x.lower() or x.lower().startswith('torch') for x in meta['project']['dependencies'])
    p=subprocess.run([sys.executable,'-m','evidence_rag','--help'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'ingest' in p.stdout and 'activate' in p.stdout

def test_container_uses_root_context_and_general_service():
    import yaml
    docker=(ROOT/'main-system/Dockerfile').read_text();compose=yaml.safe_load((ROOT/'main-system/docker-compose.yml').read_text());service=compose['services']['evidence-rag']
    assert service['build']=={'context':'..','dockerfile':'main-system/Dockerfile'}
    assert service['ports']==['127.0.0.1:8000:8000'] and service['volumes']==['evidence_data:/data']
    assert 'evidence_rag.api:app' in docker and 'server_text01:app' not in docker
    assert 'COPY evidence_rag ./evidence_rag' in docker and 'COPY . .' not in docker
    assert 'BAILIAN_API_KEY' not in (ROOT/'main-system/docker-compose.yml').read_text()
