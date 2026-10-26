
from __future__ import annotations

import copy
import IQ_recommender as shared
from .. import terminal
from .item_scorers.ollama_scorer import SCORING_PROTOCOL

QUESTION_PROTOCOL = 'pebol_candidate_aspect_zero_based_v3'
RANKING_PROTOCOL = shared.RANKING_PROTOCOL

class SimulatedUser:
    def __init__(self, calls, user):
        self.calls, self.user = calls, user

    def get_response(self, query, history):
        return shared.simulate(self.calls, self.user['private_context'], history,
                               query, self.user['catalog'])

class DialogueSimulator:
    def __init__(self, config):
        self.config = config

    def run_dialogue(self, user, pe_module, calls, saved_rounds=None, on_checkpoint=None):
        
        saved_rounds = saved_rounds or []
        if sum(r['action'] == 'ask' for r in saved_rounds) > self.config['dialogue_sim']['num_turns']:
            raise ValueError('Checkpoint exceeds the question budget')
        if pe_module.interactions:
            raise ValueError('Replay requires a fresh PEBOL module')
        history, rounds = [], []
        pe_module.excluded_questions = []
        while len(history) < self.config['dialogue_sim']['num_turns']:
            turn = len(history)
            query = pe_module.get_query()
            response = user.get_response(query, list(history))
            shared.validate_schema(response, shared.SIMULATOR_RESPONSE)
            skipped = response['answer'] == 2
            if not skipped:
                pe_module.update_from_response(query, response)
            snapshot = {'answered_questions': turn, 'history': copy.deepcopy(history),
                'action': 'skip' if skipped else 'ask', 'question': query,
                'selected_item_id': pe_module.queried_items[-1],
                'aspect': pe_module.aspects[-1], 'observed_response': response,
                'item_beliefs': [{'item_id': i, **b} for i, b in pe_module.belief.items()]}
            index = len(rounds)
            if index < len(saved_rounds) and snapshot != saved_rounds[index]:
                raise ValueError('PEBOL replay differs from the saved round')
            if skipped:
                pe_module.excluded_questions.append(copy.deepcopy(query))
            else:
                history.append({'question': query, 'answer': response['answer']})
            rounds.append(snapshot)
            if on_checkpoint:
                on_checkpoint(rounds)
        
        
        prediction = terminal.terminal_prediction(calls.cfg, calls, history, pe_module.public_catalog)
        ranking, belief = prediction['ranking'], prediction['final_preference']
        rounds.append({'answered_questions': len(history), 'history': copy.deepcopy(history),
            'action': 'stop', 'stop_reason': 'budget', 'ranking': ranking, 'belief': belief,
            **prediction,
            'acquisition_state': {'protocol': SCORING_PROTOCOL,
                'item_beliefs': [{'item_id': i, **b} for i, b in pe_module.belief.items()]}})
        return {'method': 'pebol', 'rounds': rounds, 'history': history, 'T': len(history),
            'questions_presented': len(rounds) - 1,
            'skipped_question_count': len(pe_module.excluded_questions),
            'interaction_cost': self.config['interaction_cost'] * len(history),
            'question_protocol': QUESTION_PROTOCOL, 'simulator_protocol': shared.SIMULATOR_PROTOCOL,
            'evaluation_protocol': shared.EVALUATION_PROTOCOL, 'ranking_protocol': shared.ranking_protocol(calls.cfg),
            **terminal.terminal_metadata(calls.cfg),
            'acquisition_scoring_protocol': SCORING_PROTOCOL,
            'stop_reason': 'budget', 'ranking': ranking, 'evaluation_policy': 'terminal_only',
            'preference_recovery_representation': 'shared_terminal_preference'}
