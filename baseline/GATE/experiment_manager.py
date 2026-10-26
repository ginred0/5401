"""Run the GATE yes/no adaptation with movie or music personas and saved labels."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import re
import time
from datetime import datetime, timezone

from ... import llm_recommender as shared
from .. import terminal
from .agent import GateAgent, QUESTION_PROTOCOL
from .dialogue_sim import DialogueSimulator
from .eval_manager import write_reports


ROOT = Path(__file__).resolve().parent


def load_config(common_path, gate_path, *, backbone=None, task='movie'):
    common = json.loads(common_path.read_text())
    common['task'] = task
    if backbone is not None:
        common.update({f'{role}_model': backbone['model'] for role in ('recommender', 'simulator', 'judge')})
        common['host'] = backbone['host']
    shared.validate_config(common)
    if (common.get('eig_stop_policy', 'cost') != 'cost'
            or common.get('final_preference_source', 'method') != 'method'):
        raise ValueError('EIG ablation settings do not apply to GATE')
    options = json.loads(gate_path.read_text())
    if set(options) != {'variant', 'max_skips'} or options['variant'] != 'questions_yn':
        raise ValueError('GATE supports variant=questions_yn and max_skips')
    if type(options['max_skips']) is not int or options['max_skips'] < 1:
        raise ValueError('GATE max_skips must be a positive integer')
    return common, options


def prepare(output, common, config, users, inputs, expected_users, *, backbone=None):
    if len(users) != expected_users:
        raise ValueError(f'Expected {expected_users} users, found {len(users)}')
    sources = sorted(p for p in ROOT.rglob('*') if p.is_file()
                     and 'runs' not in p.relative_to(ROOT).parts
                     and (p.suffix in {'.py', '.jinja2'} or p.name == 'UPSTREAM.json'))
    sources += [Path(shared.__file__), Path(shared.prompts.__file__),
                *(shared.ROOT / name for name in (
                    'recommender_utils.py', 'data_utils.py', 'native_qa.py',
                    'native_persona.py', 'preference_evaluation.py',
                    'openai_backend.py', 'anthropic_backend.py'))]
    sources.append(Path(terminal.__file__))
    definition = {'method': 'gate', 'common_configuration': common, 'gate_configuration': config,
        'inputs_sha256': {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in inputs.items()},
        'sources_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'dataset_signature': shared.fingerprint(users), 'expected_users': expected_users,
        'persona_ids': [u['persona_id'] for u in users], 'task': common['task'],
        'question_protocol': QUESTION_PROTOCOL, 'ranking_protocol': shared.ranking_protocol(common), **terminal.terminal_metadata(common),
        'evaluation_protocol': shared.EVALUATION_PROTOCOL}
    if backbone is not None:
        definition['backbone'] = backbone
    manifest = {'configuration': definition, 'signature': shared.fingerprint(definition)}
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        if (output / 'runtime.json').exists() or any((output / 'users').glob('*.json')):
            raise ValueError('Inputs, parameters or source changed; choose a new output directory')
    shared.write_json_atomic(path, manifest)
    shared.write_json_atomic(output / 'preflight.json', {
        'status': 'preflight_passed', 'method': 'gate', 'users': len(users),
        'backbone': backbone, 'task': common['task'], 'gate_configuration': config,
        'persona_ids': definition['persona_ids'],
        'candidate_count': common['candidate_count'], 'Tmax': common['Tmax'],
        'max_recommender_calls_per_user_before_retries': common['Tmax'] + config['max_skips'] + terminal.terminal_call_budget(common),
        'max_simulator_calls_per_user': common['Tmax'] + config['max_skips'], 'max_judge_calls_per_user': 1,
        'calls_per_user_without_skips': {'recommender': common['Tmax'] + terminal.terminal_call_budget(common), 'simulator': common['Tmax']},
        'call_budget_note': 'Answer=2 replacements add calls up to max_skips; only 0/1 enter the transcript.',
        'terminal_ranking_calls_per_user': 1,
        'ranking_protocol': shared.ranking_protocol(common), **terminal.terminal_metadata(common),
        'evaluation_protocol': shared.EVALUATION_PROTOCOL,
        'evaluation_policy': 'terminal_only',
        'results': 'No inference or empirical results are produced by prepare-only.'})
    return manifest


def execute(output, common, config, users, manifest, max_users=None, *, client_factory=None, client_factories=None):
    clients, digests = {}, {}
    for role in ('recommender', 'simulator', 'judge'):
        factory = (client_factories or {}).get(role) or client_factory or shared.OllamaClient
        client = factory(common[f'{role}_model'], common['host'], timeout=common['timeout'],
            num_ctx=common['num_ctx'], num_predict=common['num_predict'], retries=common['retries'])
        clients[role], digests[role] = client, client.check_model()
    packages = {}
    for name in ('openai', 'anthropic', 'jsonschema', 'python-dotenv'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    runtime = {'manifest_signature': manifest['signature'], 'model_digests': digests,
        'temperature': 0, 'package_versions': packages,
        'question_protocol': QUESTION_PROTOCOL, 'simulator_protocol': shared.SIMULATOR_PROTOCOL,
        'evaluation_protocol': shared.EVALUATION_PROTOCOL, 'ranking_protocol': shared.ranking_protocol(common), **terminal.terminal_metadata(common)}
    api_backends = {role: client.runtime_metadata for role, client in clients.items()
                    if hasattr(client, 'runtime_metadata')}
    if api_backends:
        runtime.update(api_backends=api_backends, temperature=None)
    path = output / 'runtime.json'
    if path.exists() and json.loads(path.read_text()) != runtime:
        raise ValueError('Model revision or runtime changed; choose a new output directory')
    shared.write_json_atomic(path, runtime)
    signature = shared.fingerprint(runtime)
    calls = shared.Calls(output / 'calls', common, clients, signature)
    records = []
    for user in users[:max_users]:
        identity = user['persona_id']
        user_key = shared.fingerprint(identity)
        path = output / 'users' / f'{user_key}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved['runtime_signature'] != signature
                    or saved['record_hash'] != shared.fingerprint(saved['record'])):
                raise ValueError('Corrupt GATE user result')
            record = saved['record']
        else:
            started = time.monotonic()
            checkpoint_path = output / 'session_checkpoints' / f'{user_key}.json'
            previous, elapsed, evaluation, attempts = [], 0., 0., 0
            started_at = datetime.now(timezone.utc).isoformat()
            if checkpoint_path.exists():
                saved = json.loads(checkpoint_path.read_text())
                if (saved['runtime_signature'] != signature or saved['persona_id'] != identity
                        or saved['checkpoint_hash'] != shared.fingerprint(saved['checkpoint'])):
                    raise ValueError('Corrupt GATE session checkpoint')
                state = saved['checkpoint']
                previous = state['rounds']
                elapsed, evaluation, attempts = state['elapsed'], state['evaluation'], state['attempts']
                started_at = state['started_at']
            latest = previous
            current_evaluation = 0.

            def checkpoint(rounds):
                nonlocal latest
                latest = copy.deepcopy(rounds)
                if rounds:
                    snapshot = rounds[-1]
                    turn_file = f'{snapshot["answered_questions"]:02d}'
                    if snapshot['action'] == 'skip':
                        turn_file += f'_skip_{len(rounds):04d}'
                    shared.write_json_atomic(output / 'rounds' / user_key / f'{turn_file}.json', snapshot)
                state = {'rounds': latest, 'elapsed': elapsed + time.monotonic() - started,
                         'evaluation': evaluation + current_evaluation,
                         'attempts': attempts + 1, 'started_at': started_at}
                shared.write_json_atomic(checkpoint_path, {'persona_id': identity,
                    'runtime_signature': signature, 'checkpoint': state,
                    'checkpoint_hash': shared.fingerprint(state)})
                shared.write_json_atomic(output / 'progress.json', {'status': 'running',
                    'method': 'gate', 'persona_id': identity,
                    'answered_questions': sum(r['action'] == 'ask' for r in rounds),
                    'questions_presented': len(rounds),
                    'skipped_question_count': sum(r['action'] == 'skip' for r in rounds),
                    'completed_users': len(records), 'expected_users': len(users)})

            agent = GateAgent(common, calls)
            try:
                record = DialogueSimulator(common, config).run_dialogue(
                    user, agent, saved_rounds=previous, on_checkpoint=checkpoint)
                final = record['rounds'][-1]
                final['recommendation_metrics'] = shared.recommendation_metrics(record['ranking'], user['target'])
                judge_started = time.monotonic()
                try:
                    final['preference_evaluation'] = shared.evaluate_preference(calls, user['reference'], final['belief'])
                finally:
                    current_evaluation += time.monotonic() - judge_started
                shared.write_json_atomic(output / 'rounds' / user_key /
                                         f'{record["T"]:02d}.json', final)
            except Exception as exc:
                checkpoint(latest)
                shared.write_json_atomic(output / 'progress.json', {'status': 'failed',
                    'persona_id': identity, 'error': str(exc), 'completed_users': len(records)})
                raise
            total = elapsed + time.monotonic() - started
            judge_total = evaluation + current_evaluation
            record.update(persona_id=identity, label_tie_count=user['label_tie_count'],
                label_insufficient_evidence_count=user['label_insufficient_evidence_count'],
                label_audit=user.get('label_audit', {}), dataset_signature=manifest['configuration']['dataset_signature'],
                started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=total - judge_total, evaluation_seconds=judge_total,
                total_elapsed_seconds=total, execution_attempts=attempts + 1)
        record.update(candidate_items=user['catalog'], target=user['target'])
        shared.write_json_atomic(path, {'runtime_signature': signature, 'record': record,
                                      'record_hash': shared.fingerprint(record)})
        records.append(record)
        write_reports(output, records, common, len(users), config)
        print(f'GATE {identity}: T={record["T"]}; completed={len(records)}/{len(users)}', flush=True)
    shared.write_json_atomic(output / 'progress.json', {'status': 'complete' if len(records) == len(users) else 'partial',
        'method': 'gate', 'completed_users': len(records), 'expected_users': len(users)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=shared.ROOT / 'llm_recommender_config.json')
    parser.add_argument('--gate-config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--task', choices=['movie', 'music'], default='movie')
    parser.add_argument('--backbone', default='deepseek-thinking-low',
                        help='deepseek-thinking-low, deepseek-thinking-disabled, claude-sonnet-low, or an Ollama model name')
    parser.add_argument('--ollama-host', default='http://localhost:11434')
    parser.add_argument('--personas', type=Path, default=shared.ROOT / 'sampled_personas.jsonl')
    parser.add_argument('--audits', type=Path, help='Accepted task labels; defaults to <task>_persona_audits.jsonl')
    parser.add_argument('--items', type=Path, help='Defaults to the selected task pool')
    parser.add_argument('--schema', type=Path, default=shared.DEFAULT_SCHEMA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-users', type=int, default=20)
    parser.add_argument('--has-not-finished-user-id-list', '--has_not_finished_user_id_list', nargs='*', default=[],
                        help='Only select these persona_id strings; an empty list uses normal sampling')
    parser.add_argument('--max-users', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.max_users is not None and args.max_users < 1:
        parser.error('--max-users must be positive')
    if args.expected_users < 1:
        parser.error('--expected-users must be positive')
    backbone = shared.resolve_backbone(args.backbone, args.ollama_host)
    common, cfg = load_config(args.config, args.gate_config, backbone=backbone, task=args.task)
    if args.items is None:
        pool = 'movies' if args.task == 'movie' else 'musical'
        args.items = shared.ROOT / 'task_dataset' / f'{pool}_sample_1000.jsonl'
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', backbone['name']).strip('._-') or 'model'
    output = args.output or shared.ROOT / 'recommender_runs' / (
        f"{safe_name}_gate_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    inputs = {k: getattr(args, k) for k in ('config', 'gate_config', 'personas', 'items', 'schema')}
    if args.audits is None and args.personas == shared.ROOT / 'sampled_personas.jsonl':
        args.audits = shared.ROOT / f'{args.task}_persona_audits.jsonl'
    if args.audits is not None:
        inputs['audits'] = args.audits
    resolved = output.resolve()
    if (any(p.resolve() == resolved or resolved in p.resolve().parents for p in inputs.values())
            or resolved == ROOT or resolved in ROOT.parents
            or (ROOT in resolved.parents and ROOT / 'runs' not in resolved.parents)):
        raise ValueError('Output must not contain inputs or overwrite GATE source; use runs/ or an external directory')
    users = shared.load_dataset(args.personas, args.items, args.schema, common,
                                audits_path=args.audits,
                                limit=None if args.has_not_finished_user_id_list else args.expected_users,
                                has_not_finished_user_id_list=args.has_not_finished_user_id_list)
    missing = set(args.has_not_finished_user_id_list) - {u['persona_id'] for u in users}
    if missing:
        raise ValueError(f'Requested user IDs have no accepted data: {sorted(missing)}')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A GATE process is already using this output directory')
        expected_users = len(users) if args.has_not_finished_user_id_list else args.expected_users
        manifest = prepare(output, common, cfg, users, inputs, expected_users, backbone=backbone)
        print(f'Prepared GATE: {len(users)} users x {common["candidate_count"]} candidates; output={output}', flush=True)
        if args.execute:
            execute(output, common, cfg, users, manifest, args.max_users,
                    client_factory=shared.backbone_client_factory(backbone))
        else:
            print('Offline preparation complete; no model backend was contacted.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
