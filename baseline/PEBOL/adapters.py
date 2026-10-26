
from __future__ import annotations

import json
from pathlib import Path

import IQ_recommender as shared
from .item_scorers.ollama_scorer import PROMPT as ITEM_SCORING_PROMPT

PROMPTS = {
    'PEBOL_ITEM_SCORES': ITEM_SCORING_PROMPT,
    'PEBOL_ASPECT': """For MOVIE recommendation, follow the supplied upstream_prompt to identify
one important aspect of the supplied movie. Use only supplied movie metadata;
do not invent cast/director metadata or other movie facts. Choose an aspect of
at most three words that is distinct from previous_aspects.

Return only JSON: aspect_value is a nonempty string.""",
    'PEBOL_QUESTION': """Based on the recommender and user question-answer interaction history,
generate exactly one question to the user, so as to understand the user's
preferences for Movie recommendations. Follow the supplied upstream_prompt:
ask a Verify yes-or-no question about the supplied aspect of the movie.
The option texts can be in natural language, rather than fixed yes/no values.
Assign the affirmative alternative to 0 and the negative alternative to 1.
Do not repeat any question already asked in history, including paraphrases.
Avoid excluded_questions as well; they provide no preference evidence.
Use only supplied movie metadata. Do not invent cast/director metadata,
undefined movie A/B placeholders, or unrelated domains.

Return only JSON: question contains id, type (Verify), question (the text),
and options mapping "0" (affirmative) and "1" (negative) to their answer texts.""",
}

class Calls(shared.Calls):
    
    def ask(self, role, stage, payload, schema, validate=None, *, system_prompt=None, user_prompt=None):
        if stage in PROMPTS:
            system_prompt = PROMPTS[stage]
            if self.cfg.get('task', 'movie') == 'music':
                system_prompt = system_prompt.replace('cast/director metadata', 'artist/genre/recording metadata')
                for old, new in (('MOVIE', 'MUSIC RECORDING'), ('Movie', 'Music recording'),
                                 ('movies', 'music recordings'), ('movie', 'music recording')):
                    system_prompt = system_prompt.replace(old, new)
                system_prompt += '\n' + shared.prompts.MUSIC_DOMAIN
        
        return super().ask(role, stage, payload, schema, validate,
                           system_prompt=system_prompt, user_prompt=user_prompt)

class CatalogLoader:
    
    def __init__(self, catalog, calls):
        self.llm = calls
        self.items = {item['item_id']: {'description': json.dumps(
            item, ensure_ascii=False, sort_keys=True)} for item in catalog}
        if len(self.items) != len(catalog) or not self.items:
            raise ValueError('Catalog must contain unique candidate IDs')

    def get_data(self):
        return self.items

class CachedScorer:
    
    def __init__(self, scorer, directory: Path, signature=None):
        self.scorer = scorer
        self.cache = {}

    def score_items(self, preference, items):
        key = json.dumps([preference, list(items.items())], sort_keys=True)
        if key not in self.cache:
            scores = self.scorer.score_items(preference, items)
            if set(scores) != set(items):
                raise ValueError('Scorer must return one probability per candidate')
            if any(not shared.number(v) or not 0 <= v <= 1 for v in scores.values()):
                raise ValueError('Invalid item-score probability')
            self.cache[key] = scores
        return dict(self.cache[key])

def make_module(config, catalog, calls, scorer):
    
    from .pe_modules.dt_pe_module import DTPEModule

    class OllamaPEModule(DTPEModule):
        
        public_catalog = catalog

        def get_aspect_val(self, item_desc):
            previous = [a['aspect_value'] for a in self.aspects]
            prompt = self.jinja_env.get_template(self.config['query']['aspect_gen_template']).render(
                item_desc=item_desc, aspects=self.aspects)

            def check(answer):
                aspect = answer['aspect_value']
                if len(aspect.split()) > 3 or shared.question_key(aspect) in {
                        shared.question_key(a) for a in previous}:
                    raise ValueError('Extract a new aspect of at most three words')

            return self.llm.ask('recommender', 'PEBOL_ASPECT', {
                'upstream_prompt': prompt, 'previous_aspects': previous},
                shared.object_schema(aspect_value=shared.TEXT), check)

        def get_query(self):
            prompt = self.get_one_item_query()
            history = [{'question': q['question_json'], 'answer': q['answer']}
                       for q in self.interactions]

            def check(answer):
                q = answer['question']
                excluded = getattr(self, 'excluded_questions', [])
                shared.check_fresh_questions([q], [*history, *({'question': q} for q in excluded)])
                if q['type'] != 'Verify':
                    raise ValueError('PEBOL uses Verify questions')
                if shared.question_key(q['options']['0']) == shared.question_key(q['options']['1']):
                    raise ValueError('Options must differ')

            payload = {'upstream_prompt': prompt, 'aspect': self.aspects[-1], 'history_json': history}
            if getattr(self, 'excluded_questions', []):
                payload['excluded_questions'] = self.excluded_questions
            return self.llm.ask('recommender', 'PEBOL_QUESTION', payload,
                shared.object_schema(question=shared.QUESTION), check)['question']

        def update_from_response(self, query, response):
            shared.validate_schema(response, shared.BINARY_RESPONSE)
            
            super().update_from_response(query['question'], 'yes' if response['answer'] == 0 else 'no')
            self.interactions[-1].update(question_json=query, answer=response['answer'])

    return OllamaPEModule(config, CatalogLoader(catalog, calls), scorer)
