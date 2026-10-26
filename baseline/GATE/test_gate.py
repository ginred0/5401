
import copy
import csv
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import IQ_recommender as shared
from . import experiment_manager as manager
from .agent import GateAgent, RANKING_PROTOCOL
from .dialogue_sim import DialogueSimulator
from . import prompts

class FakeClient:
    requests = []
    fail_second_answer = False
    fail_item = None
    skip_first = False
    skip_all = False
    invalid_score = False
    tie_scores = False

    def __init__(self, model, host, **kwargs):
        self.model = model
        self.last_attempts = []
        self.runtime_metadata = {'provider': 'offline', 'temperature': 'provider_default'}

    def check_model(self):
        return 'offline:' + self.model

    def chat_json(self, system, payload, schema, seed, validate, *, user_prompt=None):
        self.requests.append(copy.deepcopy({'system': system, 'payload': payload,
                                            'user_prompt': user_prompt}))
        if schema['type'] == 'array':
            from test_recommender_requests import judgment
            answer = [judgment('Like') for _ in payload['inferred_atoms']]
        else:
            fields = schema['properties']
            if 'question' in fields:
                n = len(payload['history_json']) + len(payload['excluded_questions'])
                subject = 'movies' if payload['task'] == 'movie' else 'music recordings'
                answer = {'question': {'id': f'q{n}', 'type': 'Verify',
                    'question': f'Do you enjoy theme{n} {subject}?',
                    'options': {'0': 'Yes', '1': 'No'}}}
            elif 'answers' in fields:
                if self.fail_second_answer and len(payload['history_json']) == 1:
                    raise RuntimeError('simulated interruption')
                qid = payload['question_json']['id']
                value = '2' if self.skip_all or (self.skip_first and qid == 'q0') else '0'
                title = fields['instrument']['properties']['title']['enum'][0]
                answer = {'instrument': {'id': 'recommender_preference_qa', 'title': title},
                          'answers': [{'questionId': qid, 'value': value}]}
            elif 'ranking' in fields:
                if self.fail_item is not None:
                    raise RuntimeError('prediction interruption')
                ids = [item['item_id'] for item in payload['candidate_items']]
                answer = {'ranking': ids if self.tie_scores else list(reversed(ids))}
                if self.invalid_score:
                    answer = {'ranking': [ids[0]] * len(ids)}
            elif any(key in fields for key in ('preferences', 'answer_irrationalities', 'joint_weights')):
                from ..test_terminal import fixture_answer
                answer = fixture_answer(payload, fields, 'Like: comedy' if 'MOVIE' in system else 'Like: jazz')
            elif 'description' in fields:
                answer = {'description': 'Like: comedy' if 'MOVIE' in system else 'Like: jazz'}
            else:
                raise AssertionError(fields)
        validate(answer)
        self.last_attempts = [{'status': 'accepted', 'usage': {'prompt_tokens': 12, 'completion_tokens': 8}}]
        return answer

class GateTests(unittest.TestCase):
    def setUp(self):
        no_network = patch.object(socket.socket, 'connect', side_effect=AssertionError('offline test'))
        no_network.start()
        self.addCleanup(no_network.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.common, self.config = manager.load_config(
            shared.ROOT / 'IQ_recommender_config.json', manager.ROOT / 'config.json')
        self.common.update(Tmax=2, candidate_count=2)
        self.config['max_skips'] = 3
        self.user = {
            'persona_id': 'offline-user',
            'catalog': [{'item_id': 123, 'title': 'Public one'},
                        {'item_id': 456, 'title': 'Public two'}],
            'target': 456,
            'private_context': shared.build_native_persona_context({
                'dimensions': {'filmg_comedy': 'Like', 'musg_jazz': 'Like',
                               'skill_logical_reasoning': 'Master'},
                'rationalization': {'settings': {'skill_logical_reasoning': 'Master'},
                                    'decision_instruction': 'private-persona-marker'}}),
            'reference': {'interest_dimensions': {'filmg_comedy': 'Like'}},
            'label_tie_count': 0, 'label_insufficient_evidence_count': 0}
        self.manifest = {'configuration': {'dataset_signature': 'fixture'}}
        FakeClient.requests = []
        for name in ('fail_second_answer', 'skip_first', 'skip_all', 'invalid_score', 'tie_scores'):
            setattr(FakeClient, name, False)
        FakeClient.fail_item = None

    def calls(self, directory=None):
        return shared.Calls(directory or self.root / 'calls', self.common,
                            {r: FakeClient('offline', 'offline') for r in ('recommender', 'simulator', 'judge')},
                            'fixture')

    def execute(self, users=None, max_users=None):
        manager.execute(self.root, self.common, self.config, users or [self.user], self.manifest,
                        max_users, client_factory=FakeClient)

    def record(self):
        return json.loads(next((self.root / 'users').glob('*.json')).read_text())['record']

    def test_native_dialogue_and_shared_eig_ranking_for_both_tasks(self):
        for task in ('movie', 'music'):
            with self.subTest(task=task):
                self.common['task'] = task
                calls = self.calls(self.root / task / 'calls')
                record = DialogueSimulator(self.common, self.config).run_dialogue(
                    self.user, GateAgent(self.common, calls))
                self.assertEqual(record['ranking'], [456, 123])
                self.assertEqual(record['T'], 2)
                self.assertEqual(record['ranking_protocol'], RANKING_PROTOCOL)
                self.assertEqual(record['rounds'][-1]['ranking_details']['ranking'], [456, 123])
                saved = [json.loads(p.read_text()) for p in calls.directory.glob('*.json')]
                simulator = [c for c in saved if c['request']['role'] == 'simulator']
                self.assertEqual(len(simulator), 2)
                self.assertIn('persona', simulator[0]['request']['payload'])
                for c in saved:
                    req = c['request']
                    if req['role'] != 'recommender':
                        continue
                    text = json.dumps(req)
                    self.assertNotIn('private-persona-marker', text)
                    self.assertNotIn('interest_dimensions', text)
                    self.assertNotIn('"target"', text)
                    self.assertNotIn('"reference"', text)
                    if req['stage'] == 'GATE_QUESTION':
                        self.assertNotIn('candidate_item', text)
                        self.assertNotIn('Public one', text)
                    if task == 'music':
                        self.assertNotIn('MOVIE', req['system'])
                        self.assertNotIn('Movie', req['system'])
                self.assertTrue(all(c.get('api_attempts') for c in saved))

    def test_skips_are_excluded_from_transcript_and_do_not_count_as_answers(self):
        FakeClient.skip_first = True
        self.execute()
        record = self.record()
        self.assertEqual((record['T'], record['questions_presented'], record['skipped_question_count']), (2, 3, 1))
        self.assertEqual([r['action'] for r in record['rounds']], ['skip', 'ask', 'ask', 'stop'])
        for req in FakeClient.requests:
            if 'M' in req['payload']:
                self.assertEqual(len(req['payload']['history_json']), 2)
                self.assertNotIn('theme0', json.dumps(req['payload']))
        self.assertEqual(record['interaction_cost'], 2 * self.common['c'])

    def test_all_skips_use_same_uniform_prior_as_eig(self):
        FakeClient.skip_all = True
        self.execute()
        record = self.record()
        self.assertEqual((record['T'], record['questions_presented'], record['stop_reason']), (0, 3, 'skip_limit'))
        self.assertEqual(record['history'], [])
        self.assertEqual(record['rounds'][-1]['belief']['joint_weights'],
                         [[1 / (self.common['M'] * self.common['L'])] * self.common['L']] * self.common['M'])
        stages = [json.loads(p.read_text())['request']['stage'] for p in (self.root / 'calls').glob('*.json')]
        self.assertNotIn('DIRECT_PREFERENCE', stages)
        self.assertIn('PREFERENCE_EVALUATION', stages)
        self.assertNotIn('WEIGHTS', stages)

    def test_interruption_replay_and_pebol_report_contract(self):
        FakeClient.fail_second_answer = True
        with self.assertRaisesRegex(RuntimeError, 'interruption'):
            self.execute()
        self.assertFalse(list((self.root / 'users').glob('*.json')))
        checkpoint = json.loads(next((self.root / 'session_checkpoints').glob('*.json')).read_text())
        self.assertEqual(len(checkpoint['checkpoint']['rounds']), 1)
        FakeClient.fail_second_answer = False
        self.execute()
        record = self.record()
        self.assertEqual(record['execution_attempts'], 2)
        self.assertEqual(sum('excluded_questions' in r['payload'] for r in FakeClient.requests), 2)
        self.assertEqual(record['rounds'][-1]['recommendation_metrics']['ndcg_at_10'], 1)
        for filename in ('manifest.json', 'preflight.json'):
            
            self.assertFalse((self.root / filename).exists())
        for filename in ('runtime.json', 'results.json', 'per_user_metrics.csv',
                         'evaluation_metrics.json', 'evaluation_metrics.md', 'evaluation_metrics.tex',
                         'progress.json'):
            self.assertTrue((self.root / filename).exists(), filename)
        self.assertEqual(len(list((self.root / 'item_score_calls').glob('*.json'))), 0)
        results = json.loads((self.root / 'results.json').read_text())
        self.assertEqual(results['method'], 'gate')
        self.assertEqual(results['users'][0]['final_recommendation']['ranking'], [456, 123])
        self.assertEqual(results['users'][0]['rounds'][0]['actual_answer'], 0)
        from ..PEBOL.eval_manager import write_reports
        comparison = self.root / 'pebol-report'
        comparison.mkdir()
        write_reports(comparison, [record], self.common, 1, {})
        with (comparison / 'per_user_metrics.csv').open() as a, (self.root / 'per_user_metrics.csv').open() as b:
            self.assertEqual(list(csv.DictReader(a)), list(csv.DictReader(b)))
        count = len(FakeClient.requests)
        self.execute()
        self.assertEqual(len(FakeClient.requests), count)

    def test_interruption_during_ranking_reuses_completed_preference_inference(self):
        FakeClient.fail_item = 456
        with self.assertRaisesRegex(RuntimeError, 'prediction interruption'):
            self.execute()
        FakeClient.fail_item = None
        self.execute()
        first_item_calls = [r for r in FakeClient.requests
                            if 'M' in r['payload']]
        self.assertEqual(len(first_item_calls), 1)
        self.assertEqual(self.record()['ranking'], [456, 123])

    def test_invalid_prediction_does_not_publish_results(self):
        FakeClient.invalid_score = True
        with self.assertRaises(ValueError):
            self.execute()
        self.assertFalse(list((self.root / 'users').glob('*.json')))
        self.assertEqual(json.loads((self.root / 'progress.json').read_text())['status'], 'failed')

    def test_shared_ranking_preserves_candidate_id_types(self):
        FakeClient.tie_scores = True
        self.assertEqual(GateAgent(self.common, self.calls()).finalize([], self.user['catalog'])['ranking'], [123, 456])
        catalog = [{'item_id': '001', 'title': 'First recording'},
                   {'item_id': '1', 'title': 'Second recording'}]
        self.assertEqual(GateAgent(self.common, self.calls()).finalize([], catalog)['ranking'], ['001', '1'])

    def test_repeated_questions_and_reversed_options_are_rejected(self):
        valid = {'id': 'q0', 'type': 'Verify', 'question': 'Do you like comedy?',
                 'options': {'0': 'Yes', '1': 'No'}}
        for history, candidate in (
                ([{'question': valid, 'answer': 0}], valid),
                ([], {**valid, 'options': {'0': 'No', '1': 'Yes'}})):
            with self.subTest(history=history), patch.object(
                    FakeClient, 'chat_json', return_value={'question': candidate}):
                with self.assertRaises(ValueError):
                    GateAgent(self.common, self.calls()).question(history, [])

    def test_invalid_gate_configuration_and_foreign_cli_options_fail(self):
        path = self.root / 'gate.json'
        for config in ({'variant': 'questions_open', 'max_skips': 3},
                       {'variant': 'questions_yn', 'max_skips': 0},
                       {'variant': 'questions_yn', 'max_skips': True}):
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                manager.load_config(shared.ROOT / 'IQ_recommender_config.json', path)
        for args in (['--method', 'gate', '--eig-stop-policy', 'budget'],
                     ['--method', 'eig', '--gate-config', str(path)],
                     ['--method', 'gate', '--pebol-config', str(path)]):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                shared.main([*args, '--prepare-only'])

    def test_tampered_session_and_completed_record_are_rejected(self):
        self.execute()
        result_path = next((self.root / 'users').glob('*.json'))
        saved = json.loads(result_path.read_text())
        saved['record']['ranking'].reverse()
        result_path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'Corrupt GATE user result'):
            self.execute()
        result_path.unlink()
        checkpoint_path = next((self.root / 'session_checkpoints').glob('*.json'))
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint['checkpoint']['rounds'][0]['observed_response']['answer'] = 1
        checkpoint_path.write_text(json.dumps(checkpoint))
        with self.assertRaisesRegex(ValueError, 'Corrupt GATE session checkpoint'):
            self.execute()

    def test_replay_checks_saved_answers_against_cached_calls(self):
        calls = self.calls()
        agent = GateAgent(self.common, calls)
        runner = DialogueSimulator(self.common, self.config)
        record = runner.run_dialogue(self.user, agent)
        rounds = copy.deepcopy(record['rounds'][:-1])
        rounds[0]['observed_response']['answer'] = 1
        with self.assertRaisesRegex(ValueError, 'replay differs'):
            runner.run_dialogue(self.user, agent, saved_rounds=rounds)

    def test_prompts_keep_paper_structure_and_answer_text(self):
        history = [{'question': {'question': 'Do you like jazz?', 'options': {'0': 'Yes', '1': 'No'}},
                    'answer': 1}]
        text = prompts.question_prompt('music', history, [])
        self.assertTrue(text.startswith('Your task is to'))
        self.assertIn('Previous questions:\n- Do you like jazz? -> No', text)
        self.assertIn('Generate the most informative yes/no question', text)
        self.assertIn('non-expert humans', text)
        self.assertIn('listening', text)
        text = prompts.prediction_prompt('music', history, self.user['catalog'][0])
        self.assertIn('probability between 0 and 1', text)
        self.assertIn('Do you like jazz? -> No', text)

    def fixture_inputs(self):
        inputs = self.root / 'inputs'
        inputs.mkdir(exist_ok=True)
        (inputs / 'common.json').write_text(json.dumps(self.common))
        (inputs / 'schema.json').write_text(json.dumps({'columns': [{'id': 'filmg_comedy'}, {'id': 'musg_jazz'}]}))
        values = {'personas.jsonl': [{'persona_id': 'offline-user',
                                    'dimensions': {'filmg_comedy': 'Like', 'musg_jazz': 'Like'}}],
                  'items.jsonl': self.user['catalog'],
                  'audits.jsonl': [{'persona_id': 'offline-user', 'status': 'accepted',
                                    'candidates': [123, 456], 'target': 456}]}
        for name, rows in values.items():
            (inputs / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return ['--config', str(inputs / 'common.json'), '--personas', str(inputs / 'personas.jsonl'),
                '--audits', str(inputs / 'audits.jsonl'), '--items', str(inputs / 'items.jsonl'),
                '--schema', str(inputs / 'schema.json'), '--expected-users', '1']

    def test_main_cli_offline_preflight_and_explicit_id_validation(self):
        args = self.fixture_inputs()
        with patch.object(shared, 'backbone_client_factory', side_effect=AssertionError('must stay offline')):
            for task in ('movie', 'music'):
                for backbone in ('deepseek-thinking-low', 'claude-sonnet-low', 'qwen3:8b'):
                    output = self.root / f'{task}-{backbone.replace(":", "_")}'
                    self.assertEqual(shared.main([
                        '--method', 'gate', '--task', task, '--backbone', backbone, *args,
                        '--output', str(output), '--prepare-only',
                        '--has-not-finished-user-id-list', 'offline-user']), 0)
                    manifest = json.loads((output / 'manifest.json').read_text())['configuration']
                    self.assertEqual(manifest['task'], task)
                    self.assertEqual(manifest['persona_ids'], ['offline-user'])
                    self.assertTrue((output / 'preflight.json').exists())
                    self.assertFalse((output / 'runtime.json').exists())
            with self.assertRaisesRegex(ValueError, 'no accepted data'):
                shared.main(['--method', 'gate', *args, '--prepare-only',
                             '--has-not-finished-user-id-list', 'offline-user', 'missing'])

    def test_main_cli_executes_by_default(self):
        args = self.fixture_inputs()
        with patch.object(shared, 'backbone_client_factory', return_value=FakeClient):
            self.assertEqual(shared.main(['--method', 'gate', *args, '--output', str(self.root / 'run')]), 0)
        self.assertTrue((self.root / 'run' / 'results.json').exists())

    def test_music_explicit_input_resolution_and_output_naming(self):
        with patch.object(shared, 'load_dataset', return_value=[self.user]) as load, \
                patch.object(manager, 'prepare', return_value=self.manifest) as prepare, \
                patch.object(shared, 'ROOT', self.root):
            (self.root / 'IQ_recommender_config.json').write_text(json.dumps(self.common))
            shared.main(['--method', 'gate', '--task', 'music', '--items', str(self.root / 'music.jsonl'), '--prepare-only', '--expected-users', '1'])
        self.assertEqual(load.call_args.args[1], self.root / 'music.jsonl')
        self.assertEqual(load.call_args.kwargs['audits_path'], self.root / 'music_persona_audits.jsonl')
        self.assertEqual(load.call_args.args[3]['task'], 'music')
        self.assertIn('_gate_music_', prepare.call_args.args[0].name)

    def test_changed_inputs_cannot_resume_after_inference(self):
        inputs = {'gate_config': manager.ROOT / 'config.json'}
        manager.prepare(self.root, self.common, self.config, [self.user], inputs, 1)
        shared.write_json_atomic(self.root / 'runtime.json', {'fixture': True})
        changed = copy.deepcopy(self.user)
        changed['target'] = 123
        with self.assertRaisesRegex(ValueError, 'changed'):
            manager.prepare(self.root, self.common, self.config, [changed], inputs, 1)

    def test_partial_status_for_limited_execution(self):
        other = {**self.user, 'persona_id': 'second-user'}
        self.execute([self.user, other], max_users=1)
        results = json.loads((self.root / 'results.json').read_text())
        self.assertEqual((results['status'], results['completed_users'], results['expected_users']), ('partial', 1, 2))

if __name__ == '__main__':
    unittest.main()
