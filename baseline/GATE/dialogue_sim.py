
from __future__ import annotations

import copy

import IQ_recommender as shared
from .. import terminal
from .agent import QUESTION_PROTOCOL

class DialogueSimulator:
    def __init__(self, common, config):
        self.common, self.config = common, config

    def run_dialogue(self, user, agent, saved_rounds=None, on_checkpoint=None):
        saved_rounds = saved_rounds or []
        history, rounds, excluded = [], [], []
        if any(r['action'] not in {'ask', 'skip'} for r in saved_rounds):
            raise ValueError('GATE checkpoints must contain interaction rounds only')
        while len(history) < self.common['Tmax'] and len(excluded) < self.config['max_skips']:
            question = agent.question(history, excluded)
            response = shared.simulate(
                agent.calls, user['private_context'], history, question, user['catalog'])
            shared.validate_schema(response, shared.SIMULATOR_RESPONSE)
            skipped = response['answer'] == 2
            snapshot = {'answered_questions': len(history), 'history': copy.deepcopy(history),
                        'action': 'skip' if skipped else 'ask', 'question': question,
                        'observed_response': response}
            index = len(rounds)
            if index < len(saved_rounds) and snapshot != saved_rounds[index]:
                raise ValueError('GATE replay differs from the saved round')
            if skipped:
                excluded.append(copy.deepcopy(question))
            else:
                history.append({'question': question, 'answer': response['answer']})
            rounds.append(snapshot)
            if on_checkpoint:
                on_checkpoint(rounds)
        if len(rounds) < len(saved_rounds):
            raise ValueError('GATE checkpoint exceeds configured budgets')
        reason = 'budget' if len(history) == self.common['Tmax'] else 'skip_limit'
        prediction = agent.finalize(history, user['catalog'])
        ranking, belief = prediction['ranking'], prediction['final_preference']
        rounds.append({'answered_questions': len(history), 'history': copy.deepcopy(history),
                       'action': 'stop', 'stop_reason': reason, 'ranking': ranking,
                       **prediction, 'belief': belief})
        return {'method': 'gate', 'task': self.common['task'], 'variant': self.config['variant'],
                'rounds': rounds, 'history': history, 'T': len(history),
                'questions_presented': len(rounds) - 1, 'skipped_question_count': len(excluded),
                'interaction_cost': self.common['c'] * len(history),
                'question_protocol': QUESTION_PROTOCOL, 'ranking_protocol': shared.ranking_protocol(self.common),
                **terminal.terminal_metadata(self.common),
                'simulator_protocol': shared.SIMULATOR_PROTOCOL,
                'evaluation_protocol': shared.EVALUATION_PROTOCOL,
                'stop_reason': reason, 'ranking': ranking, 'evaluation_policy': 'terminal_only',
                'preference_recovery_representation': 'shared_terminal_preference'}
