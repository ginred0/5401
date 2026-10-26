
from __future__ import annotations

import json

import IQ_recommender as shared
from .item_scorer import ItemScorer

SCORING_PROTOCOL = 'pebol_ollama_compatibility_probability_v1'
PROMPT = """For MOVIE recommendation, assess every candidate item by compatibility
with the supplied preference evidence. Use only the supplied candidate_items
metadata and stated preference evidence. Do not invent user tastes or
cast/director metadata. The evidence can express a positive aspect or a dislike
(prefixed with not).

For each item, estimate the probability that the user would like it under this
evidence. Use 0.5 when the evidence does not distinguish liking from disliking.
These are independent item utilities, not a distribution over candidate_items:
do not normalize them to sum to one. Return probabilities directly; do not
output ranks or transform positions into probabilities.

Return only JSON: item_scores is an array with exactly one record per supplied
candidate, in input order. Each record contains item_id and a finite
probability in [0,1]."""

class OllamaScorer(ItemScorer):
    
    def __init__(self, config, calls):
        super().__init__(config)
        self.calls = calls
        backend = getattr(calls.clients['recommender'], 'runtime_metadata', {})
        self.metadata = {'protocol': SCORING_PROTOCOL, 'role': 'recommender',
                         'model': calls.cfg['recommender_model'], 'temperature': backend.get('temperature', 0)}

    def score_items(self, preference, items):
        catalog = [json.loads(item['description']) for item in items.values()]
        ids = list(items)
        if (not ids or type(ids[0]) not in (int, str)
                or any(type(i) is not type(ids[0]) for i in ids)
                or [item['item_id'] for item in catalog] != ids):
            raise ValueError('Candidate IDs must match the public catalog and share an integer/string type')
        id_schema = {'type': 'integer' if type(ids[0]) is int else 'string', 'enum': ids}
        score = shared.object_schema(item_id=id_schema, probability=shared.PROBABILITY)

        def check(answer):
            if [s['item_id'] for s in answer['item_scores']] != ids:
                raise ValueError('Return every candidate ID exactly once in input order')

        answer = self.calls.ask('recommender', 'PEBOL_ITEM_SCORES', {
            'preference_json': {'id': 'pebol_feedback', 'description': preference},
            'candidate_items': catalog},
            shared.object_schema(item_scores=shared.array_schema(score, len(ids))), check)
        return {s['item_id']: s['probability'] for s in answer['item_scores']}
