"""TREC ndcg_cut linear relevance gain; BEIR binary recall/MRR and ID filtering."""
import math

def evaluate(qid,qrels,ranking):
    # BEIR EvaluateRetrieval ignores identical query/doc IDs in results, not qrels.
    ids=[x for x in dict.fromkeys(ranking) if x!=qid]
    relevant={doc:grade for doc,grade in qrels.items() if grade>0}
    if not relevant: raise ValueError('qrel has no positive documents')
    gains=[relevant.get(doc,0) for doc in ids[:10]]
    ideal=sorted(relevant.values(),reverse=True)[:10]
    dcg=sum(g/math.log2(i+2) for i,g in enumerate(gains))
    idcg=sum(g/math.log2(i+2) for i,g in enumerate(ideal))
    hits=sum(doc in relevant for doc in ids[:20])
    first=next((i+1 for i,doc in enumerate(ids[:10]) if doc in relevant),None)
    return {'ndcg_at_10':dcg/idcg,'recall_at_20':hits/len(relevant),'mrr_at_10':1/first if first else 0.0,
            'relevant_count':len(relevant),'relevant_hit_at_20':hits,'first_relevant_rank':first}
