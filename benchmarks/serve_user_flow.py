import argparse
import json
import os
from pathlib import Path
import sys

import uvicorn

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    config=json.load(sys.stdin)
    if not isinstance(config,dict) or not config.get('auth'):
        raise ValueError('Explicit private service configuration required on stdin')
    allowed={'DB','AUTH','OV_URL','OV_SCOPE','OV_TIMEOUT','RERANKER','RERANK_DEVICE',
             'RERANK_DTYPE','RERANK_TIMEOUT','RERANK_BATCH_SIZE','TOKENIZER','TOKENIZER_REVISION',
             'MODEL_URL','MODEL_NAME','MODEL_KEY','MODEL_TIMEOUT','MODEL_TEMPERATURE','MODEL_THINKING','MODEL_REASONING_EFFORT','MODEL_CONTEXT_WINDOW',
             'FRAMING_RESERVE','OUTPUT_RESERVE','MAX_OUTPUT_TOKENS','RERANK_THRESHOLD',
             'STORE_CONVERSATIONS','STREAM_INCLUDE_USAGE','GENERAL_SYSTEM_PROMPT','DEFAULT_EVIDENCE_ITEMS'}
    for name in allowed:
        value=config.get(name.lower())
        if value is not None:
            os.environ['EVIDENCE_RAG_'+name]=json.dumps(value) if name=='AUTH' else str(value)
    uvicorn.run('evidence_rag.api:app',host='127.0.0.1',port=args.port,log_level='warning',access_log=False)

if __name__=='__main__':
    main()
