"""GATE's prompted question policy with shared EIG terminal inference.

Adapted from GenerativeQuestionsAgent and BaseActiveLearningAgent in the
author repository (MIT). No EIG calculation or PEBOL acquisition is used.
"""
from __future__ import annotations

from ... import llm_recommender as shared
from .. import terminal
from . import prompts

QUESTION_PROTOCOL = 'gate_questions_yn_zero_based_v1'
RANKING_PROTOCOL = shared.RANKING_PROTOCOL


class GateAgent:
    def __init__(self, common, calls):
        self.common, self.calls = common, calls
        self.task = common['task']
        prompts.domain(self.task)

    def question(self, history, excluded):
        def check(answer):
            q = answer['question']
            shared.check_fresh_questions(
                [q], [*history, *({'question': old} for old in excluded)])
            if q['type'] != 'Verify' or q['options'] != {'0': 'Yes', '1': 'No'}:
                raise ValueError('GATE requires Verify with 0=Yes, 1=No')

        return self.calls.ask(
            'recommender', 'GATE_QUESTION',
            {'task': self.task, 'history_json': history, 'excluded_questions': excluded},
            shared.object_schema(question=shared.QUESTION), check,
            system_prompt=prompts.question_prompt(self.task, history, excluded))['question']

    def finalize(self, history, catalog):
        return terminal.terminal_prediction(self.common, self.calls, history, catalog)
