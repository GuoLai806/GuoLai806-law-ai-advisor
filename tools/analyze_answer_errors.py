"""Describe disagreements against unchanged annotations; never override gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def analyze(report_path, output_path):
    payload = Path(report_path).read_bytes()
    report = json.loads(payload)
    rows = report['rows']
    if len(rows) != report['queries_expected'] or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('A complete, unique query set is required')
    labels = ('SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO')
    slices = {}
    for label in labels:
        selected = [r for r in rows if r['gold_label'] == label]
        correct = sum(not r.get('failure') and r.get('predicted_label') == label for r in selected)
        slices[label] = {
            'queries': len(selected), 'correct': correct,
            'accuracy': correct / len(selected) if selected else None,
            'evaluation_failures': sum(bool(r.get('failure')) for r in selected),
            'predictions': dict(Counter('INVALID' if r.get('failure') else r.get('predicted_label')
                                       for r in selected)),
        }
    disagreements = []
    positive = [r for r in rows if r.get('gold_documents')]
    covered = [r for r in positive if r.get('final_evidence_recall') == 1.0]
    for row in rows:
        if not row.get('failure') and row.get('predicted_label') == row['gold_label']:
            continue
        evidence = row.get('evidence', [])
        disagreements.append({
            'query_id': row['query_id'], 'gold_label': row['gold_label'],
            'predicted_label': row.get('predicted_label'), 'failure': row.get('failure'),
            'user_flow_completed': row.get('user_flow_completed'),
            'final_evidence_recall': row.get('final_evidence_recall'),
            'all_annotated_evidence_documents_available': bool(row.get('gold_documents'))
                and row.get('final_evidence_recall') == 1.0,
            'gold_documents': row.get('gold_documents', []),
            'selected_documents': [r.get('document_id') for r in evidence],
            'cited_documents': row.get('cited_documents', []),
            'selected_reranker_scores': [r.get('reranker_score') for r in evidence],
            'user_total_ms': row.get('user_total_ms'),
        })
    result = {
        'input_sha256': hashlib.sha256(payload).hexdigest(), 'queries': len(rows),
        'label_slices': slices, 'disagreements': disagreements,
        'positive_claims': len(positive), 'all_gold_documents_in_selected_evidence': len(covered),
        'positive_wrong_label_despite_complete_gold_document_coverage': sum(
            not r.get('failure') and r.get('predicted_label') != r['gold_label'] for r in covered),
        'positive_evaluation_failure_despite_complete_gold_document_coverage': sum(
            bool(r.get('failure')) for r in covered),
        'scope': 'Descriptive analysis against existing, unchanged claim/document annotations; '
            'document coverage is not semantic answer support; no expert adjudication or gold repair; '
            'the observed formal disagreements must not be used to tune this run.',
    }
    with Path(output_path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'disagreements'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.report, args.output)
