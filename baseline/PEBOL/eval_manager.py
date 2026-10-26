
from __future__ import annotations

import csv
import IQ_recommender as shared
from .. import terminal
from .dialogue_sim import QUESTION_PROTOCOL
from .item_scorers.ollama_scorer import SCORING_PROTOCOL

METRICS = (*shared.RECOMMENDATION_METRICS, 'preference_elicitation_accuracy',
           'T', 'interaction_cost', 'skipped_question_count', 'questions_presented',
           *shared.TIMING_METRICS)

def report_round(snapshot, number):
    question = snapshot.get('question')
    response = snapshot.get('observed_response')
    answer = response['answer'] if response is not None else None
    return {**snapshot, 'round': number, 'selected_question': question,
            'actual_answer': answer,
            'presented_questions': ([{'question': question, 'actual_answer': answer,
                                     'answer_text': question['options'].get(
                                         str(answer), 'Insufficient evidence')}]
                                    if question is not None and response is not None else [])}

def write_reports(output, records, common, expected_users, pebol_config):
    rows = []
    for record in records:
        final = record['rounds'][-1]
        rows.append({'persona_id': record['persona_id'],
            'T': record['T'], 'interaction_cost': record['interaction_cost'],
            'stop_reason': record['stop_reason'],
            'questions_presented': record['questions_presented'],
            'skipped_question_count': record['skipped_question_count'],
            'label_tie_count': record['label_tie_count'],
            'label_insufficient_evidence_count': record['label_insufficient_evidence_count'],
            **{k: record[k] for k in shared.TIMING_METRICS},
            **final['recommendation_metrics'],
            **{k: final['preference_evaluation'][k] for k in ('preference_elicitation_accuracy', 'inferred_atoms')},
        })
    summary = {'method': 'pebol', 'task': common.get('task', 'movie'),
        'status': 'complete' if len(records) == expected_users else 'partial',
        'completed_users': len(records), 'expected_users': expected_users,
        'candidate_count': common['candidate_count'], 'evaluation_policy': 'terminal_only',
        'question_protocol': QUESTION_PROTOCOL, 'simulator_protocol': shared.SIMULATOR_PROTOCOL,
        'evaluation_protocol': shared.EVALUATION_PROTOCOL, **terminal.terminal_report_metadata(records, common),
        'acquisition_scoring_protocol': SCORING_PROTOCOL,
        'terminal': {k: shared.mean_present(row[k] for row in rows) for k in METRICS} if rows else {},
        'preference_score_samples': sum(row['preference_elicitation_accuracy'] is not None for row in rows),
        'preference_score_unresolved_users': sum(row['preference_elicitation_accuracy'] is None for row in rows),
        'timing_definition': 'Dialogue, shared terminal preference inference and ranking in elapsed_seconds; '
            'shared atomic judging in evaluation_seconds. Active replay and cache lookup time included; '
            'model loading, downtime and report export excluded.',
        'label_audit': {'users_with_random_final_tie': sum(
            r.get('label_audit', {}).get('final_tie_resolved_randomly', False) for r in records),
            'policy': 'Keep all saved users and tournament labels; no relabeling.'},
        'uncertainty': 'No confidence interval or superiority claim is produced by this export.'}
    catalog_users = []
    for record, row in zip(records, rows):
        catalog = record.get('candidate_items', [])
        target = record.get('target')
        catalog_users.append({'persona_id': record['persona_id'],
            'rounds': [report_round(snapshot, index) for index, snapshot in enumerate(record['rounds'], 1)],
            'final_recommendation': {
                'candidates': catalog, 'ranking': record['ranking'], 'target': target,
                'target_item': next((item for item in catalog if item['item_id'] == target), None)},
            'metrics': {key: value for key, value in row.items() if key != 'persona_id'}})
    overall = {**summary['terminal'],
        'total_interaction_cost': sum(record['interaction_cost'] for record in records),
        'total_answered_questions': sum(record['T'] for record in records)}
    shared.write_json_atomic(output / 'results.json', {
        'schema_version': 4, 'status': summary['status'], 'method': 'pebol',
        'configuration': common, 'pebol_configuration': pebol_config,
        'completed_users': len(records), 'expected_users': expected_users,
        'metric_definitions': {
            'hr_at_5': '1 if the reference target is in the first 5 ranked items, otherwise 0.',
            'hr_at_10': '1 if the reference target is in the first 10 ranked items, otherwise 0.',
            'ndcg_at_5': 'Single-target discounted gain within the first 5 ranked items.',
            'ndcg_at_10': 'Single-target discounted gain within the first 10 ranked items.',
            'preference_elicitation_accuracy': 'v12 relevance-weighted atom alignment of the shared terminal preference.',
            'interaction_cost': 'c * T; T counts informative 0/1 answers. This is not API billing.',
            'actual_answer': '0 or 1 selects the corresponding option; 2 means insufficient evidence.',
            'overall': 'Means across completed users, except fields explicitly prefixed total_.'},
        'overall': overall, 'users': catalog_users, 'evaluation_summary': summary})
    shared.write_json_atomic(output / 'evaluation_metrics.json', summary)
    if rows:
        with (output / 'per_user_metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# PEBOL terminal evaluation', '',
        f'Status: {summary["status"]}; users: {len(records)}/{expected_users}.', '',
        '| Metric | Mean |', '| --- | ---: |']
    lines.extend(f'| {k} | {shared.format_metric(v, 6)} |' for k, v in summary['terminal'].items())
    lines.extend(['', 'HR@5/10 and NDCG@5/10 use the same single saved target as the main method.',
        'The legacy-named preference_elicitation_accuracy uses v12 relevance-weighted level alignment, '
        'normalized within each inferred atom and averaged across inferred atoms. '
        f'Preference scores available: {summary["preference_score_samples"]}/{len(rows)}; '
        f'unresolved users: {summary["preference_score_unresolved_users"]}.',
        'PEBOL has item utility beliefs rather than narrative hypotheses: recovery is measured on a '
        'shared terminal preference. The same summary feeds the main method Agent.rank and atomic judge. '
        'Beta utilities affect acquisition only, not terminal ranking.',
        'PEBOL uses its fixed Tmax budget; the main EIG policy may stop early. Compare T, cost and time '
        'alongside ranking quality. Equal maximum questions do not imply equal inference compute.', '',
        'Combined report: results.json (rounds, answers, candidates, targets, per-user and overall metrics).',
        'Additional exports: per_user_metrics.csv, evaluation_metrics.json/tex and users/*.json.', ''])
    (output / 'evaluation_metrics.md').write_text('\n'.join(lines))
    tex = [f'% PEBOL terminal evaluation; status={summary["status"]}; N={len(records)}/{expected_users}',
           f'\\noindent Evaluation status: {summary["status"]}; completed users: '
           f'{len(records)}/{expected_users}; candidate items per user: {common["candidate_count"]}.\\par',
           '\\begin{tabular}{rrrrr}',
           'HR@5 & HR@10 & NDCG@5 & NDCG@10 & Preference alignment \\\\',
           '\\hline']
    if rows:
        tex.append(' & '.join(shared.format_metric(summary['terminal'][key]) for key in
            (*shared.RECOMMENDATION_METRICS, 'preference_elicitation_accuracy')) + ' \\\\')
    tex.append('\\end{tabular}')
    if rows:
        tex.append(f'\\par Mean questions: {summary["terminal"]["T"]:.4f}; '
                   f'mean interaction cost: {summary["terminal"]["interaction_cost"]:.4f}.\\par')
    (output / 'evaluation_metrics.tex').write_text('\n'.join(tex) + '\n')
