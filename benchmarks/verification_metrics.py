import re

LABELS=('SUPPORT','CONTRADICT','NOT_ENOUGH_INFO')

def gold_label(claim):
    labels={item['label'] for evidence in claim.get('evidence',{}).values() for item in evidence}
    if not labels:
        return 'NOT_ENOUGH_INFO'
    if len(labels)!=1 or not labels.issubset(set(LABELS[:2])):
        raise ValueError('Ambiguous gold claim labels')
    return next(iter(labels))

def parse_verdict(answer):
    match=re.match(r'^\s*(?:\*\*)?(SUPPORT|CONTRADICT|NOT_ENOUGH_INFO)\b',answer)
    return match.group(1) if match else None

def aggregate(rows):
    if not rows:
        raise ValueError('No evaluation rows')
    confusion={gold:{prediction:0 for prediction in (*LABELS,'INVALID')} for gold in LABELS}
    pair_gold=pair_predicted=pair_correct=0
    for row in rows:
        gold=row['gold_label']
        predicted=None if row.get('failure') else row.get('predicted_label')
        confusion[gold][predicted if predicted in LABELS else 'INVALID']+=1
        expected=set(row.get('gold_documents',[]))
        actual=set(row.get('cited_documents',[])) if predicted in LABELS[:2] else set()
        pair_gold+=len(expected)
        pair_predicted+=len(actual)
        pair_correct+=len(expected & actual) if predicted==gold else 0
    f1={}
    for label in LABELS:
        true=confusion[label][label]
        predicted=sum(confusion[gold][label] for gold in LABELS)
        expected=sum(confusion[label].values())
        f1[label]=2*true/(predicted+expected) if predicted+expected else 0.0
    precision=pair_correct/pair_predicted if pair_predicted else 0.0
    recall=pair_correct/pair_gold if pair_gold else 0.0
    return {'claim_label_accuracy':sum(confusion[label][label] for label in LABELS)/len(rows),
            'claim_label_macro_f1':sum(f1.values())/len(LABELS),'label_f1':f1,'confusion':confusion,
            'cited_document_label_precision':precision,'cited_document_label_recall':recall,
            'cited_document_label_f1':2*precision*recall/(precision+recall) if precision+recall else 0.0,
            'cited_document_label_counts':{'correct':pair_correct,'predicted':pair_predicted,'gold':pair_gold},
            'scope':'custom claim-label and cited-document+label metrics; not official sentence-rationale leaderboard scoring'}
