
from __future__ import annotations

import argparse
import copy
import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from functools import partial
from statistics import mean
from typing import Any

if __package__:
    from . import IQ_recommender_prompts as prompts
    from .data_utils import DEFAULT_SCHEMA, fingerprint, label_audit, read_jsonl, schema_fields, write_json_atomic
    from .recommender_utils import OllamaClient, build_persona_context, item_context, stable_seed
    from .native_persona import build_native_persona_context, SIMULATOR_PROTOCOL
    from .preference_evaluation import reference_atoms, score_atom_judgments
else:
    import IQ_recommender_prompts as prompts
    from data_utils import DEFAULT_SCHEMA, fingerprint, label_audit, read_jsonl, schema_fields, write_json_atomic
    from recommender_utils import OllamaClient, build_persona_context, item_context, stable_seed
    from native_persona import build_native_persona_context, SIMULATOR_PROTOCOL
    from preference_evaluation import reference_atoms, score_atom_judgments

ROOT = Path(__file__).resolve().parent
TIMING_METRICS = ("elapsed_seconds", "evaluation_seconds", "total_elapsed_seconds")
QUESTION_METRICS = ("questions_presented",)
RECOMMENDATION_METRICS = ("hr_at_5", "hr_at_10", "ndcg_at_5", "ndcg_at_10")
QUESTION_PROTOCOL = "movie_context_zero_based_options_v8"
EVALUATION_PROTOCOL = "relevance_weighted_atom_alignment_v12"
RANKING_PROTOCOL = "map_preference_v2"
REQUEST_PROTOCOL = "per_hypothesis_irrationality_vectors_v3"
BACKBONE_PRESETS = {
    'deepseek-thinking-low': {'provider': 'deepseek', 'model': 'deepseek-flash',
        'host': 'https://api.deepseek.com', 'thinking': 'enabled', 'reasoning_effort': 'low'},
    'deepseek-thinking-disabled': {'provider': 'deepseek', 'model': 'deepseek-flash',
        'host': 'https://api.deepseek.com', 'thinking': 'disabled', 'reasoning_effort': 'low'},
    'claude-sonnet-low': {'provider': 'anthropic', 'model': 'claude-sonnet-5-5',
        'host': 'https://api.anthropic.com', 'thinking': 'adaptive', 'reasoning_effort': 'low'},
}


def resolve_backbone(name: str, ollama_host: str = 'http://localhost:11434') -> dict:
    """Resolve named API presets; all other names are literal Ollama model IDs."""
    name = name.strip()
    if not name:
        raise ValueError('Backbone must be nonempty')
    alias = re.sub(r'[\s_]+', '-', name.casefold())
    if alias == 'claude-sonet-low':
        alias = 'claude-sonnet-low'
    if alias in BACKBONE_PRESETS:
        return {'name': alias, **BACKBONE_PRESETS[alias]}
    return {'name': name, 'provider': 'ollama', 'model': name, 'host': ollama_host}


def backbone_client_factory(backbone: dict):
    """Build a client factory only at execution time; never persist credentials."""
    settings = {}

    from dotenv import dotenv_values
   
    if backbone['provider'] == 'ollama':
        return OllamaClient
    if backbone['provider'] == 'deepseek':
        settings.update(dotenv_values(".local/.env.local"))
        api_key = settings.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError('Set DEEPSEEK_API_KEY or OPENAI_API_KEY to execute the DeepSeek backbone')
        if __package__:
            from .openai_backend import OpenAIClient as client
        else:
            from openai_backend import OpenAIClient as client
    else:
        settings.update(dotenv_values(".local/.env.local"))
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError('Set ANTHROPIC_API_KEY to execute the Claude backbone')
        if __package__:
            from .anthropic_backend import AnthropicClient as client
        else:
            from anthropic_backend import AnthropicClient as client
    options = {'connection_retries': 3} if backbone['provider'] == 'deepseek' else {}
    return partial(client, api_key=api_key, thinking=backbone['thinking'],
                   reasoning_effort=backbone['reasoning_effort'], streaming=True, **options)


def question_protocol(cfg):
    return QUESTION_PROTOCOL if cfg.get('task', 'movie') == 'movie' else 'music_recordings_zero_based_options_v2'


def ranking_protocol(cfg):
    return ('history_summary_v1' if cfg.get('final_preference_source', 'method') == 'history'
            else RANKING_PROTOCOL)


def request_protocol(cfg):
    return (REQUEST_PROTOCOL if cfg.get('enable_irrationality', True)
            else 'preference_only_vectors_v1')


def experiment_metadata(cfg):
    """Keep legacy runtime signatures intact when no ablation is enabled."""
    if not any(key in cfg for key in ('eig_stop_policy', 'final_preference_source', 'enable_irrationality')):
        return {}
    metadata = {'eig_stop_policy': cfg.get('eig_stop_policy', 'cost'),
                'final_preference_source': cfg.get('final_preference_source', 'method')}
    if 'enable_irrationality' in cfg:
        metadata['enable_irrationality'] = cfg['enable_irrationality']
    if metadata['final_preference_source'] == 'history':
        metadata['final_summary_prompt_sha256'] = fingerprint(
            prompts.for_task('DIRECT_PREFERENCE', cfg.get('task', 'movie')))
    return metadata


def number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def mean_present(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def format_metric(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def object_schema(**properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def array_schema(items: dict, size: int | None = None) -> dict:
    result = {"type": "array", "items": items}
    if size is not None:
        result.update(minItems=size, maxItems=size)
    return result


TEXT = {"type": "string", "minLength": 1}
BINARY = {"type": "integer", "enum": [0, 1]}
BINARY_RESPONSE = object_schema(answer=BINARY)
SIMULATOR_RESPONSE = object_schema(answer={"type": "integer", "enum": [0, 1, 2]})
PROBABILITY = {"type": "number", "minimum": 0, "maximum": 1}
PROFILE = object_schema(id=TEXT, description=TEXT)
QUESTION = object_schema(id=TEXT, type={"type": "string", "enum": ["Verify", "Compare"]},
                         question=TEXT, options=object_schema(**{"0": TEXT, "1": TEXT}))
QUESTION_OUTPUT = object_schema(**{key: value for key, value in QUESTION['properties'].items() if key != 'id'})


def question_input(question: dict) -> dict:
    """Send question semantics, keeping local IDs out of model requests."""
    return {key: question[key] for key in ('type', 'question', 'options')}


def history_input(history: list[dict]) -> list[dict]:
    """Retain every observed answer and option mapping, without trace metadata."""
    return [{'question': question_input(entry['question']), 'answer': entry['answer']}
            for entry in history]


def question_record(question: dict) -> dict:
    return {'id': 'Q' + fingerprint(question)[:16], **question}


def preference_profiles(descriptions: list[str], prefix: str = 'P') -> list[dict]:
    profiles = [{'id': f'{prefix}{i+1}', 'description': text} for i, text in enumerate(descriptions)]
    distinct(profiles, 'description')
    if prefix == 'P':
        for profile in profiles:
            preference_atoms(profile['description'])
    return profiles


def preference_atoms(description: str) -> list[str]:
    """Split labeled clauses locally; preserve free-text meaning and qualifications."""
    if not isinstance(description, str):
        raise ValueError('Preference description must be a string')
    atoms, seen = [], set()
    for clause in re.split(r'[;；\n]+', description):
        clause = clause.strip()
        if not clause:
            continue
        match = re.fullmatch(r'(?:(Tentative)\s+)?(Love|Like|Neutral|Dislike|Prefer|Unknown)\s*[:：]\s*(.+)',
                             clause, flags=re.IGNORECASE)
        if not match:
            raise ValueError('Use semicolon-separated preferences: Love/Like/Neutral/Dislike/Prefer/Unknown: taste; '
                             'an optional Tentative prefix is allowed. Do not include unlabeled narrative')
        tentative, level, taste = match.groups()
        if level.casefold() == 'unknown':
            continue
        statement = ('Tentative ' if tentative else '') + level.capitalize() + ': ' + taste.strip()
        key = ' '.join(unicodedata.normalize('NFKC', statement).casefold().split())
        if key not in seen:
            seen.add(key)
            atoms.append(statement)
    return atoms


class RepeatedQuestionError(ValueError):
    """A generated question must be regenerated before it can reach a user."""


def question_key(text: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def check_fresh_questions(questions: list[dict], history: list[dict]) -> None:
    asked = {question_key(entry["question"]["question"]) for entry in history}
    def alternatives(q):
        if q.get("type") == "Compare" and isinstance(q.get("options"), dict):
            return frozenset(question_key(value) for value in q["options"].values())
        return None
    asked_pairs = {alternatives(entry["question"]) for entry in history} - {None}
    seen = set()
    seen_pairs = set()
    for question in questions:
        key = question_key(question["question"])
        pair = alternatives(question)
        if key in asked or key in seen or (pair is not None and pair in asked_pairs | seen_pairs):
            location = "this user's answered history or skipped questions" if key in asked or pair in asked_pairs else "this candidate pool"
            raise RepeatedQuestionError(
                f"Question already occurs in {location}: {json.dumps(question['question'])}. "
                "Generate a different question about a new preference. Changing its ID, capitalization, "
                "punctuation, whitespace, or merely paraphrasing does not fix repetition. "
                "Return the complete corrected JSON object")
        seen.add(key)
        if pair is not None:
            seen_pairs.add(pair)


def is_repeated_question_error(error: Exception) -> bool:
    while error is not None:
        if isinstance(error, RepeatedQuestionError):
            return True
        error = error.__cause__
    return False


def validate_schema(value: Any, schema: dict, where: str = "output") -> None:
    """Validate the schema subset used here, even when a test/client ignores format."""
    kind = schema["type"]
    if isinstance(kind, list):
        for variant in kind:
            try:
                validate_schema(value, {**schema, 'type': variant}, where)
                return
            except ValueError:
                pass
        raise ValueError(f'{where}: expected one of {kind}')
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "null": value is None,
             "string": isinstance(value, str), "boolean": type(value) is bool,
             "integer": type(value) is int, "number": number(value)}[kind]
    if not valid:
        raise ValueError(f"{where}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where}: invalid enum")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"{where}: unexpected/missing fields")
        for key, child in schema["properties"].items():
            validate_schema(value[key], child, f"{where}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError(f"{where}: wrong array size")
        for i, child in enumerate(value):
            validate_schema(child, schema["items"], f"{where}[{i}]")
        if schema.get("uniqueItems"):
            duplicates = []
            for i, child in enumerate(value):
                if child in value[:i] and child not in duplicates:
                    duplicates.append(child)
            if duplicates:
                message = f"{where}: duplicate items {json.dumps(duplicates)}."
                allowed = schema["items"].get("enum")
                if allowed is not None:
                    missing = [item for item in allowed if item not in value]
                    message += (f" Missing allowed IDs: {json.dumps(missing)}."
                                " Return every allowed ID exactly once in ranked order.")
                raise ValueError(message)
    elif kind == "string" and len(value.strip()) < schema.get("minLength", 0):
        raise ValueError(f"{where}: empty text")
    elif kind in {"number", "integer"}:
        if not number(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError(f"{where}: out of range")


def distinct(records: list[dict], text_key: str, *, with_ids: bool = True) -> None:
    for key in (("id", text_key) if with_ids else (text_key,)):
        seen = {}
        for i, record in enumerate(records):
            normalized = record[key].strip().casefold()
            if normalized in seen:
                first = seen[normalized]
                if key == "description":
                    instruction = ("Generate a materially different hypothesis consistent with the history; "
                                   "changing only its ID or capitalization does not fix a repeated description")
                elif key == "id":
                    instruction = "Assign a different unique ID to each entry"
                else:
                    instruction = "Generate a different question; whitespace or capitalization changes do not fix repetition"
                raise ValueError(f"Duplicate {key} at entries {first+1} and {i+1}: "
                                 f"{json.dumps(record[key])}. {instruction}. Return the complete corrected array")
            seen[normalized] = i


def behavior_profiles(descriptions: list[dict]) -> list[dict]:
    """Accept generated IDs, then normalize local identifiers by description order."""
    profiles = [{"id": f"I{i+1}", "description": item["description"]}
                for i, item in enumerate(descriptions)]
    distinct(profiles, "description")
    return profiles


def preference_distribution(belief: dict) -> list[dict]:
    preferences = belief['preferences']
    if 'preference_marginals' in belief:
        weights = belief['preference_marginals']
    elif 'joint_weights' in belief:
        weights = [sum(row) for row in normalize_weights(belief['joint_weights'])]
    elif len(preferences) == 1:
        weights = [1.0]
    else:
        raise ValueError('Multiple preference hypotheses require posterior probabilities')
    if len(weights) != len(preferences) or any(not number(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError('Invalid preference probabilities')
    total = sum(weights)
    return [{**p, 'probability': w/total} for p, w in zip(preferences, weights)]


def most_likely_preference(belief: dict) -> dict:
    """Select by preference marginal probability; ties keep original input order."""
    distribution = preference_distribution(belief)
    index = max(range(len(distribution)), key=lambda i: distribution[i]['probability'])
    selected = distribution[index]
    return {**selected, 'id': selected.get('id', f'P{index+1}')}


def reject_schema_conditioned_trace(trace: dict) -> None:
    if trace.get("genre_recovery", False):
        raise ValueError("This checkpoint/result used schema-conditioned questions. "
                         "Preserve it for audit; resume from a clean checkpoint or use a new output directory")


def normalize_weights(weights: list[list[float]]) -> list[list[float]]:
    if not weights or not weights[0] or any(len(row) != len(weights[0]) for row in weights):
        raise ValueError("Invalid weight dimensions")
    if any(not number(x) or x < 0 for row in weights for x in row):
        raise ValueError("Weights must be finite and nonnegative")
    
    scale = max(x for row in weights for x in row)
    if scale <= 0:
        raise ValueError("At least one joint weight must be positive")
    total = math.fsum(x / scale for row in weights for x in row)
    return [[(x / scale) / total for x in row] for row in weights]


def normalize_preference_weights(weights: list[float]) -> list[float]:
    return normalize_weights([weights])[0]


def binary_entropy(p: float) -> float:
    if not number(p) or not 0 <= p <= 1:
        raise ValueError("Probability outside [0,1]")
    return 0.0 if p in (0, 1) else -p * math.log2(p) - (1-p) * math.log2(1-p)


def preference_eig(weights: list[list[float]], ideal: list[int],
                   epsilon: list[list[float]]) -> dict:
   
    w = normalize_weights(weights)
    m, l = len(w), len(w[0])
    if len(ideal) != m or len(epsilon) != m or any(len(row) != l for row in epsilon):
        raise ValueError("Ideal/distortion dimensions differ from weights")
    if any(type(a) is not int or a not in (0, 1) for a in ideal):
        raise ValueError("Ideal answers must be binary integers")
    if any(not number(e) or not 0 <= e <= 1 for row in epsilon for e in row):
        raise ValueError("Invalid distortion probability")
    p = [[e + (1-2*e)*ideal[i] for e in row] for i, row in enumerate(epsilon)]
    v = [math.fsum(row) for row in w]
    r_theta = [math.fsum(w[i][j]*p[i][j] for j in range(l))/v[i] if v[i] else 0.0
               for i in range(m)]
    r = min(1.0, max(0.0, math.fsum(v[i]*r_theta[i] for i in range(m))))
    eig = binary_entropy(r) - math.fsum(v[i]*binary_entropy(min(1.0, max(0.0, r_theta[i])))
                                         for i in range(m) if v[i] > 0)
    if eig < -1e-12:
        raise ArithmeticError("Negative preference information gain")
    return {"eig_bits": max(0.0, eig), "preference_marginals": v,
            "conditional_answer_one": p, "answer_one_given_preference": r_theta,
            "answer_one": r}


def preference_only_eig(weights: list[float], ideal: list[int], epsilon: list[float]) -> dict:
    """I(Theta; A) with a single answer probability per preference hypothesis."""
    v = normalize_preference_weights(weights)
    m = len(v)
    validate_schema(ideal, array_schema(BINARY, m))
    validate_schema(epsilon, array_schema(PROBABILITY, m))
    r_theta = [e + (1 - 2*e)*a for a, e in zip(ideal, epsilon)]
    r = min(1.0, max(0.0, math.fsum(w*p for w, p in zip(v, r_theta))))
    eig = binary_entropy(r) - math.fsum(w*binary_entropy(p) for w, p in zip(v, r_theta))
    if eig < -1e-12:
        raise ArithmeticError('Negative preference information gain')
    return {'eig_bits': max(0.0, eig), 'preference_marginals': v,
            'conditional_answer_one': r_theta, 'answer_one_given_preference': r_theta,
            'answer_one': r}



@contextmanager
def request_progress(client, status, path, started, interval=10):
    """Show long requests even before the first stream chunk; never log model text."""
    stop, lock, latest = threading.Event(), threading.Lock(), {}
    previous = getattr(client, 'on_progress', None)
    supports_progress = hasattr(client, 'on_progress')
    label = f"[{status['role']}/{status['stage']}]"

    def update(event):
        fields = ('attempt', 'request_output_budget', 'output_characters', 'reasoning_characters', 'stream_chunks')
        with lock:
            if event.get('attempt') != latest.get('attempt'):
                latest.clear()
            latest.update({key: event[key] for key in fields if key in event})
        if previous:
            previous(event)

    def pulse():
        with lock:
            progress = dict(latest)
        elapsed = time.monotonic() - started
        phase = ('receiving_output' if progress.get('output_characters') else
                 'thinking' if progress.get('reasoning_characters') else 'waiting_api')
        write_json_atomic(path, {**status, **progress, 'phase': phase, 'elapsed_seconds': elapsed,
            'updated_at': datetime.now(timezone.utc).isoformat()})
        print(f"{label} {elapsed:.0f}s, attempt {progress.get('attempt', 1)}, {phase}; "
              f"output {progress.get('output_characters', 0)} characters", flush=True)

    def heartbeat():
        while not stop.wait(interval):
            pulse()

    if supports_progress:
        client.on_progress = update
    worker = threading.Thread(target=heartbeat, name='recommender-request-progress', daemon=True)
    print(f'{label} requesting...', flush=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()
        if supports_progress:
            client.on_progress = previous


def response_was_truncated(error):
    while error is not None:
        if str(error).endswith(('API response was truncated', 'Claude response stopped: max_tokens')):
            return True
        error = error.__cause__
    return False


def ranking_correction(answer: Any, catalog: list[dict]) -> str | None:
    """Explain an invalid permutation using only public candidate identifiers."""
    ranking = answer.get('ranking') if isinstance(answer, dict) else None
    if not isinstance(ranking, list) or any(type(item) not in (str, int) for item in ranking):
        return None
    ids = [item['item_id'] for item in catalog]
    missing = [item for item in ids if item not in ranking]
    unexpected = list(dict.fromkeys(item for item in ranking if item not in ids))
    duplicates = list(dict.fromkeys(item for i, item in enumerate(ranking) if item in ranking[:i]))
    if len(ranking) == len(ids) and not missing and not unexpected and not duplicates:
        return None
    details = json.dumps({'expected_count': len(ids), 'received_count': len(ranking),
                          'missing_ids': missing, 'unexpected_ids': unexpected,
                          'duplicate_ids': duplicates, 'allowed_ids': ids}, ensure_ascii=False)
    return (f'Ranking must be a complete candidate permutation. {details}. '
            'Copy each allowed item_id exactly once, preserving every character including zeros. '
            'Remove unexpected and repeated IDs; place missing candidates according to the preference. '
            'Return the entire corrected ranking as JSON, not just the changed IDs. '
            'The allowed_ids list is the candidate inventory, not a suggested ranking.')


class Calls:
    """Content-addressed validated calls; interrupted runs reuse every accepted call."""
    def __init__(self, directory: Path, cfg: dict, clients: dict, signature: str):
        self.directory, self.cfg, self.clients, self.signature = directory, cfg, clients, signature

    def ask(self, role: str, stage: str, payload: dict, schema: dict, validate=None,
            *, system_prompt: str | None = None, user_prompt: str | None = None) -> dict | list:
        if stage == 'SIMULATOR' and (system_prompt is None or user_prompt is None):
            raise ValueError('Simulator requests must use the native MatrAIx QA runner prompts')
        request = {"run_signature": self.signature, "role": role, "stage": stage,
                   "system": system_prompt if system_prompt is not None else prompts.for_task(stage, self.cfg.get('task', 'movie')),
                   "payload": payload, "schema": schema}
        if user_prompt is not None:
            request['user_prompt'] = user_prompt
        key = fingerprint(request)
        seed = stable_seed(self.cfg["seed"], key)
        path = self.directory / f"{key}.json"

        def check(answer):
            try:
                validate_schema(answer, schema)
                if validate:
                    validate(answer)
            except ValueError as exc:
                if stage == 'RANK' and 'candidate_items' in payload:
                    correction = ranking_correction(answer, payload['candidate_items'])
                    if correction:
                        raise ValueError(f'{exc}. {correction}') from exc
                raise

        if path.exists():
            old = json.loads(path.read_text())
            if old["request"] != request or old["seed"] != seed or old["answer_hash"] != fingerprint(old["answer"]):
                raise ValueError("Call checkpoint is corrupt")
            check(old["answer"])
            return old["answer"]
        started = time.monotonic()
        status = {"status": "requesting", "role": role, "stage": stage, "request_key": key,
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        write_json_atomic(self.directory.parent / "call_progress.json", status)
        failure_path = self.directory.parent / "failed_calls" / f"{key}.json"
        client = self.clients[role]
        client.recovery_messages = []
        if failure_path.exists():
            previous_failure = json.loads(failure_path.read_text())
            if previous_failure.get("request") == request and previous_failure.get("seed") == seed:
                client.recovery_messages = previous_failure.get("recovery_messages", [])
                if stage == 'RANK':
                    for message in reversed(client.recovery_messages):
                        if message.get('role') != 'assistant':
                            continue
                        try:
                            previous_answer = json.loads(message['content'])
                        except (ValueError, TypeError, KeyError):
                            break
                        try:
                            check(previous_answer)
                        except ValueError as exc:
                            client.recovery_messages.append({'role': 'user', 'content': f'Invalid output: {exc}'})
                        break
                if any(a.get('finish_reason') in {'length', 'max_tokens'}
                       for a in previous_failure.get('api_attempts', [])):
                    client.recovery_messages.append({'role': 'user', 'content':
                        'API response was truncated. Return concise JSON matching the schema; no explanations.'})
        regenerations = 0
        api_attempts = []
        while True:
            try:
                with request_progress(client, status, self.directory.parent / 'call_progress.json', started):
                    answer = client.chat_json(request["system"], payload, schema, seed, check,
                        **({'user_prompt': user_prompt} if user_prompt is not None else {}))
                api_attempts.extend(copy.deepcopy(getattr(client, 'last_attempts', [])))
                check(answer)
                break
            except Exception as exc:
                api_attempts.extend(copy.deepcopy(getattr(client, 'last_attempts', [])))
                recovery = getattr(client, "last_failed_messages", [])
                write_json_atomic(failure_path, {"request": request, "seed": seed, "error": str(exc),
                    "recovery_messages": recovery,
                    "api_attempts": api_attempts,
                    "updated_at": datetime.now(timezone.utc).isoformat()})
                if stage in ("QUESTIONS", "DIRECT_QUESTION") and is_repeated_question_error(exc):
                    regenerations += 1
                    client.recovery_messages = recovery
                    write_json_atomic(self.directory.parent / "call_progress.json", {
                        **status, "question_regenerations": regenerations, "last_error": str(exc),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.monotonic()-started})
                    continue
                write_json_atomic(self.directory.parent / "call_progress.json", {
                    **status, "status": "failed", "error": str(exc), "elapsed_seconds": time.monotonic()-started,
                    'updated_at': datetime.now(timezone.utc).isoformat()})
                print(f"[{role}/{stage}] failed after {time.monotonic()-started:.1f}s: {exc}", flush=True)
                raise
        saved = {"request": request, "seed": seed, "answer": answer,
                 "answer_hash": fingerprint(answer), "elapsed_seconds": time.monotonic()-started}
        if api_attempts:
            saved['api_attempts'] = api_attempts
        write_json_atomic(path, saved)
        write_json_atomic(self.directory.parent / "call_progress.json", {
            **status, "status": "accepted", "elapsed_seconds": time.monotonic()-started,
            'updated_at': datetime.now(timezone.utc).isoformat()})
        print(f'[{role}/{stage}] accepted in {time.monotonic()-started:.1f}s', flush=True)
        return answer


class Agent:
    """Questions use task context and history; ranking receives the public catalog."""
    def __init__(self, cfg: dict, calls: Calls):
        self.cfg, self.calls = cfg, calls
        self.last_ranking_details = None
        self.excluded_questions = []
        self.scoring_batch_limits = {}

    def question_history(self, history: list) -> list:
        return [*history, *({"question": q} for q in self.excluded_questions)]

    method = "iq"
    belief_for_decision = True

    def decision(self, history: list, belief: dict) -> dict:
        scored = self.score_questions(belief, self.questions(history))
        best = max(range(len(scored)), key=lambda i: scored[i]["eig_bits"])
        result = {"candidates": scored, "selected_index": best}
        if (self.cfg.get('eig_stop_policy', 'cost') == 'cost'
                and scored[best]["eig_bits"] <= self.cfg["c"] and len(history) >= 1):
            return {**result, "action": "stop", "stop_reason": "eig_not_above_cost"}
        return {**result, "action": "ask"}

    def summarize_history(self, history: list[dict]) -> dict:
        """Shared terminal summary: only actual informative Q/A is supplied."""
        summary = self.calls.ask("recommender", "DIRECT_PREFERENCE", {"history_json": history_input(history)},
                                 object_schema(description=TEXT), lambda x: preference_atoms(x['description']))
        return {"preferences": [{"id": "direct_summary", **summary}], "map_index": 0,
                "representation": "single_summary_no_belief_distribution"}

    def belief(self, history: list[dict]) -> dict:
        cfg = self.cfg
        compact_history = history_input(history)
        descriptions = self.calls.ask("recommender", "PREFERENCE", {"history_json": compact_history, "M": cfg["M"]},
            object_schema(preferences=array_schema(TEXT, cfg["M"])),
            lambda x: preference_profiles(x["preferences"]))["preferences"]
        preferences = preference_profiles(descriptions)
        if not cfg.get('enable_irrationality', True):
            if not history:
                weights = [1 / cfg['M']] * cfg['M']
            else:
                weights = self.calls.ask('recommender', 'PREFERENCE_WEIGHTS', {
                    'history_json': compact_history, 'preferences': descriptions},
                    object_schema(preference_weights=array_schema(
                        {'type': 'number', 'minimum': 0}, cfg['M'])),
                    lambda answer: normalize_preference_weights(answer['preference_weights']))['preference_weights']
            marginals = normalize_preference_weights(weights)
            return {'preferences': preferences, 'preference_marginals': marginals,
                    'map_index': max(range(cfg['M']), key=marginals.__getitem__)}
        descriptions = self.calls.ask("recommender", "BEHAVIOR", {"history_json": compact_history, "L": cfg["L"]},
            object_schema(answer_irrationalities=array_schema(TEXT, cfg["L"])),
            lambda x: preference_profiles(x["answer_irrationalities"], 'I'))["answer_irrationalities"]
        behaviors = preference_profiles(descriptions, 'I')
        if not history:
            uniform = 1/(cfg["M"]*cfg["L"])
            weights = [[uniform for _ in range(cfg["L"])] for _ in range(cfg["M"])]
        else:
            weights = self.calls.ask("recommender", "WEIGHTS", {
                "history_json": compact_history, "preferences": [p['description'] for p in preferences],
                "answer_irrationalities": descriptions}, object_schema(joint_weights=array_schema(
                    array_schema({"type": "number", "minimum": 0}, cfg["L"]), cfg["M"])),
                lambda answer: normalize_weights(answer["joint_weights"]))["joint_weights"]
        weights = normalize_weights(weights)
        marginals = [math.fsum(row) for row in weights]
        return {"preferences": preferences, "answer_irrationalities": behaviors, "joint_weights": weights,
                "preference_marginals": marginals, "map_index": max(range(cfg["M"]), key=marginals.__getitem__)}

    def questions(self, history: list[dict]) -> list[dict]:
        accepted, rejected = [], []
        attempt = 0
        while len(accepted) < self.cfg["K"]:
            missing = self.cfg["K"] - len(accepted)
            payload = {"history_json": history_input(history), "K": missing}
            if self.excluded_questions:
                payload["excluded_questions"] = [question_input(q) for q in self.excluded_questions]
            if attempt:
                payload.update(retained_questions=[question_input(q) for q in accepted],
                               rejected_questions=rejected[-self.cfg["K"]:],
                               generation_attempt=attempt)
            batch = self.calls.ask("recommender", "QUESTION_REFILL" if attempt else "QUESTIONS", payload,
                object_schema(questions=array_schema(QUESTION_OUTPUT, missing)))["questions"]
            for generated in batch:
                q = question_record(generated)
                try:
                    validate_schema(q, QUESTION)
                    check_fresh_questions([q], self.question_history(history) +
                                          [{"question": old} for old in accepted])
                    distinct([*accepted, q], "question")
                    if q["options"]["0"].strip().casefold() == q["options"]["1"].strip().casefold():
                        raise ValueError("Options must differ")
                except ValueError as exc:
                    reason = ('Repeated question or comparison alternatives; ask about a new preference'
                              if isinstance(exc, RepeatedQuestionError) else str(exc))
                    rejected.append({"question": question_input(q), "reason": reason})
                else:
                    accepted.append(q)
            attempt += 1
            write_json_atomic(self.calls.directory.parent / "question_pool_progress.json", {
                "status": "complete" if len(accepted) == self.cfg["K"] else "refilling",
                "target_count": self.cfg["K"], "accepted_count": len(accepted),
                "remaining_count": self.cfg["K"] - len(accepted),
                "generation_batches": attempt, "rejected_count": len(rejected),
                "accepted_questions": accepted,
                "updated_at": datetime.now(timezone.utc).isoformat()})
        return accepted

    def score(self, belief: dict, question: dict) -> dict:
        return self.score_questions(belief, [question])[0]

    def _question_batches(self, stage, count, request, start=0):
        if not count:
            return []
        limit = self.scoring_batch_limits.get(stage, count)
        if count > limit:
            rows = []
            for offset in range(start, start + count, limit):
                rows.extend(self._question_batches(stage, min(limit, start + count - offset), request, offset))
            return rows
        try:
            return request(start, start + count)
        except Exception as exc:
            if count <= 1 or not response_was_truncated(exc):
                raise
            self.scoring_batch_limits[stage] = max(1, count // 2)
            print(f'[{stage}] truncated batch of {count} questions; retrying in batches of '
                  f'{self.scoring_batch_limits[stage]}', flush=True)
            return self._question_batches(stage, count, request, start)

    def distortion(self, belief: dict, questions: list[dict], ideal: list[list[int]]) -> list:
        if not self.cfg.get('enable_irrationality', True):
            return self.preference_distortion(belief, questions, ideal)
        k, m, l = len(questions), len(belief['preferences']), len(belief['answer_irrationalities'])
        validate_schema(ideal, array_schema(array_schema(BINARY, m), k))
        epsilon = [[[0.0 for _ in range(l)] for _ in range(m)] for _ in range(k)]
        if not questions:
            return epsilon
        for i, preference in enumerate(belief['preferences']):
            for j, behavior in enumerate(belief['answer_irrationalities']):
                print(f'[DISTORTION] preference {i+1}/{m}, behavior {j+1}/{l}', flush=True)
                def request(start, end):
                    return self.calls.ask('recommender', 'DISTORTION', {
                        'questions': [question_input(q) for q in questions[start:end]],
                        'preference': preference['description'],
                        'ideal_answers': [row[i] for row in ideal[start:end]],
                        'answer_irrationality': behavior['description']},
                        object_schema(epsilon=array_schema(PROBABILITY, end-start)))['epsilon']
                values = self._question_batches('DISTORTION', k, request)
                for q, value in enumerate(values):
                    epsilon[q][i][j] = value
        return epsilon

    def preference_distortion(self, belief: dict, questions: list[dict], ideal: list[list[int]]) -> list:
        k, m = len(questions), len(belief['preferences'])
        validate_schema(ideal, array_schema(array_schema(BINARY, m), k))
        epsilon = [[0.0 for _ in range(m)] for _ in range(k)]
        for i, preference in enumerate(belief['preferences']):
            def request(start, end):
                return self.calls.ask('recommender', 'PREFERENCE_DISTORTION', {
                    'questions': [question_input(q) for q in questions[start:end]],
                    'preference': preference['description'],
                    'ideal_answers': [row[i] for row in ideal[start:end]]},
                    object_schema(epsilon=array_schema(PROBABILITY, end-start)))['epsilon']
            values = self._question_batches('PREFERENCE_DISTORTION', k, request)
            for q, value in enumerate(values):
                epsilon[q][i] = value
        return epsilon

    def score_questions(self, belief: dict, questions: list[dict]) -> list[dict]:
        """Score all questions per preference, then per preference/behavior pair."""
        if not questions:
            return []
        k, m = len(questions), len(belief['preferences'])
        ideal = [[0 for _ in range(m)] for _ in range(k)]
        for i, preference in enumerate(belief['preferences']):
            print(f'[IDEAL] preference {i+1}/{m}', flush=True)
            def request(start, end):
                return self.calls.ask('recommender', 'IDEAL', {
                    'questions': [question_input(q) for q in questions[start:end]],
                    'preference': preference['description']},
                    object_schema(ideal_answers=array_schema(BINARY, end-start)))['ideal_answers']
            values = self._question_batches('IDEAL', k, request)
            for q, value in enumerate(values):
                ideal[q][i] = value
        epsilon = self.distortion(belief, questions, ideal)
        if not self.cfg.get('enable_irrationality', True):
            return [{'question': question, 'ideal_answers': ideal[i], 'epsilon': epsilon[i],
                     **preference_only_eig(belief['preference_marginals'], ideal[i], epsilon[i])}
                    for i, question in enumerate(questions)]
        return [{'question': question, 'ideal_answers': ideal[i], 'epsilon': epsilon[i],
                 **preference_eig(belief['joint_weights'], ideal[i], epsilon[i])}
                for i, question in enumerate(questions)]

    def rank_hypothesis(self, preference: dict, catalog: list[dict]) -> list:
        ids = [item["item_id"] for item in catalog]
        if (not ids or type(ids[0]) not in (int, str)
                or any(type(item) is not type(ids[0]) for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Candidate IDs must be nonempty, unique and uniformly integer or string")
        def check(answer):
            if len(answer["ranking"]) != len(ids) or set(answer["ranking"]) != set(ids):
                raise ValueError("Ranking must be a complete candidate permutation")
        id_schema = {"type": "integer" if type(ids[0]) is int else "string"}
        ranking_schema = {**array_schema(id_schema, len(ids)), "uniqueItems": True}
        return self.calls.ask("recommender", "RANK", {
            "preference": preference['description'], "candidate_items": catalog},
            object_schema(ranking=ranking_schema), check)["ranking"]

    def rank(self, belief: dict, catalog: list[dict]) -> list:
        selected = most_likely_preference(belief)
        profile = {k: v for k, v in selected.items() if k != 'probability'}
        ranking = self.rank_hypothesis(profile, catalog)
        self.last_ranking_details = {'protocol': ranking_protocol(self.cfg),
                                     'selected_preference': selected, 'ranking': ranking}
        return ranking


class DirectLLMAgent(Agent):
    method = "direct_llm"
    belief_for_decision = False

    def belief(self, history: list[dict]) -> dict:
        return super().belief(history)

    def decision(self, history: list, belief: dict) -> dict:
        def check(answer):
            q = question_record(answer["question"])
            check_fresh_questions([q], self.question_history(history))
            if q["options"]["0"].strip().casefold() == q["options"]["1"].strip().casefold():
                raise ValueError("Options must differ")
        payload = {"history_json": history_input(history)}
        if self.excluded_questions:
            payload["excluded_questions"] = [question_input(q) for q in self.excluded_questions]
        question = self.calls.ask("recommender", "DIRECT_QUESTION", payload,
                                   object_schema(question=QUESTION_OUTPUT), check)["question"]
        question = question_record(question)
        return {"candidates": [{"question": question}], "selected_index": 0, "action": "ask"}


METHODS = {"iq": Agent, "direct_llm": DirectLLMAgent}


def recommendation_metrics(ranking: list, target: Any) -> dict:
    if len(ranking) != len(set(ranking)) or target not in ranking:
        raise ValueError("Invalid ranking/target")
    rank = ranking.index(target) + 1
    result = {"target_rank": rank}
    for k in (5, 10):
        hit = int(rank <= k)
        result[f"hr_at_{k}"] = hit
        result[f"ndcg_at_{k}"] = hit/math.log2(rank+1)
    return result


ABSOLUTE_PREFERENCE_SCORES = {
    'Love': {'Love': 1, 'Like': .5, 'Neutral': 0, 'Dislike': -1},
    'Like': {'Love': .5, 'Like': 1, 'Neutral': 0, 'Dislike': -.5},
    'Neutral': {'Love': 0, 'Like': 0, 'Neutral': 1, 'Dislike': 0},
    'Dislike': {'Love': -1, 'Like': -.5, 'Neutral': 0, 'Dislike': 1},
}


def evaluate_preference(calls: Calls, reference: dict | list[str], inferred: dict) -> dict:
    """Match multiple references per inferred atom; weight level scores locally."""
    belief = inferred if 'preferences' in inferred else {'preferences': [inferred]}
    selected = most_likely_preference(belief)
    atoms = preference_atoms(selected['description'])
    references = reference_atoms(reference)
    def validate_judgments(judgments):
        score_atom_judgments(atoms, judgments, references, ABSOLUTE_PREFERENCE_SCORES)
    judgments = calls.ask('judge', 'PREFERENCE_EVALUATION', {
        'reference_atoms': references, 'inferred_atoms': atoms},
        array_schema(object_schema(
            matches=array_schema(object_schema(
                reference_index={'type': 'integer', 'minimum': 0},
                relevance={'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1}))), len(atoms)),
        validate_judgments) if atoms else []
    scored = score_atom_judgments(atoms, judgments, references, ABSOLUTE_PREFERENCE_SCORES)
    return {**scored,
        'inferred_atoms': len(atoms), 'inferred_checklist': atoms,
        'atom_scores': [{'atom_id': f'A{i+1}', 'statement': atom, 'score': score}
                        for i, (atom, score) in enumerate(zip(atoms, scored['scores']))],
        'judge_decisions': judgments,
        'reference_atoms': references, 'selected_preference': selected,
        'scoring_protocol': EVALUATION_PROTOCOL}


def simulate(calls: Calls, persona: dict, history: list, question: dict, catalog: list) -> dict:
    if (persona.get('rendering_protocol') != SIMULATOR_PROTOCOL
            or not isinstance(persona.get('persona_system_prompt'), str)
            or not persona['persona_system_prompt'].strip()):
        raise ValueError('Simulator requires a MatrAIx native rendered persona; rebuild the dataset context')
    if __package__:
        from .native_qa import answer_question
    else:
        from native_qa import answer_question
    return answer_question(calls, persona, history, question, catalog)


def uninformative_questions(rounds: list[dict]) -> list[dict]:
    """Recover exclusions from completed pools, including partially answered pools."""
    questions = []
    for snapshot in rounds:
        rejected = snapshot.get("rejected_candidates")
        if rejected is not None:
            questions.extend(snapshot["candidates"][entry["candidate_index"]]["question"]
                             for entry in rejected)
        elif snapshot["action"] == "skip":
            questions.append(snapshot["candidates"][snapshot["selected_index"]]["question"])
    return questions


def session(agent: Agent, catalog: list[dict], answer_fn, on_round=None,
            resume_state=None, on_checkpoint=None) -> dict:
    """Checkpoint completed rounds; replay an interrupted round from its start."""
    agent.last_ranking_details = None
    state = {"history": [], "rounds": []}
    if resume_state is not None:
        state["rounds"] = copy.deepcopy(resume_state["rounds"])
        completed_answers = sum(r["action"] == "ask" for r in state["rounds"])

        state["history"] = copy.deepcopy(resume_state["history"][:completed_answers])
    history, rounds = state["history"], state["rounds"]
    for entry in history:
        validate_schema({"answer": entry["answer"]}, BINARY_RESPONSE)
    skipped = uninformative_questions(rounds)
    agent.excluded_questions = copy.deepcopy(skipped)
    def checkpoint():
        if on_checkpoint:
            on_checkpoint(state)
    def result(snapshot):
        record = {"method": agent.method, "rounds": rounds, "history": history, "T": len(history),
                "questions_presented": len(history) + len(skipped),
                "skipped_question_count": len(skipped),
                "interaction_cost": agent.cfg["c"]*len(history),
                "question_protocol": question_protocol(agent.cfg),
                "simulator_protocol": SIMULATOR_PROTOCOL,
                'preference_protocol': prompts.PREFERENCE_PROTOCOL,
                'request_protocol': request_protocol(agent.cfg),
                'evaluation_protocol': EVALUATION_PROTOCOL, 'ranking_protocol': ranking_protocol(agent.cfg),
                **experiment_metadata(agent.cfg),
                "stop_reason": snapshot["stop_reason"], "ranking": snapshot["ranking"],
                "evaluation_policy": "terminal_only"}
        if skipped:
            record.update(skipped_questions=copy.deepcopy(skipped), skipped_question_count=len(skipped))
        return record
    checkpoint()
    if rounds and rounds[-1]["action"] == "stop":
        return result(rounds[-1])
    while len(history) <= agent.cfg["Tmax"]:
        answered = len(history)
        history_summary = agent.cfg.get('final_preference_source', 'method') == 'history'

        need_belief = agent.belief_for_decision and not (history_summary and answered == agent.cfg['Tmax'])
        belief = agent.belief(history) if need_belief else None
        snapshot = {"answered_questions": answered, "history": list(history),
                    "questions_presented": answered + len(skipped)}
        if answered == agent.cfg["Tmax"]:
            snapshot.update(candidates=[], selected_index=None, action="stop", stop_reason="budget")
        else:
            snapshot.update(agent.decision(history, belief))
        new_entry = None
        if snapshot["action"] == "ask":
            question = snapshot["candidates"][snapshot["selected_index"]]["question"]
            try:
                check_fresh_questions([question], agent.question_history(history))
            except RepeatedQuestionError:
                continue
            snapshot["rejected_candidates"] = []
            remaining = set(range(len(snapshot["candidates"])))
            while remaining:
                index = snapshot["selected_index"]
                question = snapshot["candidates"][index]["question"]
                response = answer_fn(list(history), question)
                validate_schema(response, SIMULATOR_RESPONSE)
                snapshot["observed_response"] = response
                snapshot["questions_presented"] += 1
                if response["answer"] != 2:
                    new_entry = {"question": question, "answer": response["answer"]}
                    break
                snapshot["rejected_candidates"].append(
                    {"candidate_index": index, "observed_response": response})
                remaining.remove(index)
                if remaining:
                    snapshot["selected_index"] = max(sorted(remaining),
                        key=lambda i: snapshot["candidates"][i].get("eig_bits", 0))
                else:
                    snapshot.update(action="skip", skip_reason="all_candidates_insufficient_evidence")
        if snapshot["action"] == "stop":
            if history_summary:
                final_preference = agent.summarize_history(history)
                snapshot['final_preference'] = final_preference

                if belief is None:
                    belief = final_preference
            else:
                if belief is None:
                    belief = agent.belief(history)
                final_preference = belief
            snapshot["ranking"] = agent.rank(final_preference, catalog)
            if agent.last_ranking_details is not None:
                snapshot['ranking_details'] = agent.last_ranking_details
        if belief is not None:
            snapshot["belief"] = belief
        if on_round:
            on_round(snapshot)
        if new_entry is not None:
            history.append(new_entry)
        rejected = uninformative_questions([snapshot])
        skipped.extend(copy.deepcopy(rejected))
        agent.excluded_questions.extend(copy.deepcopy(rejected))
        rounds.append(snapshot)
        checkpoint()
        if snapshot["action"] == "stop":
            return result(snapshot)
    raise AssertionError("Session did not terminate")


def load_dataset(personas_path: Path, items_path: Path, schema_path: Path, cfg: dict, *,
                 audits_path: Path | None = None, limit: int | None = None,
                 has_not_finished_user_id_list: list[str] | None = None) -> list[dict]:
    """Join accepted audit labels to active personas, keeping audit file order.

    A nonempty unfinished-ID list restricts the cohort before limiting it; an
    empty list leaves normal sampling unchanged. Without audits_path, retain
    support for existing combined labeled datasets.
    """
    task = cfg.get('task', 'movie')
    if task not in {'movie', 'music'}:
        raise ValueError('task must be movie or music')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('User limit must be a positive integer')
    items = list(read_jsonl(items_path))
    pool = {item["item_id"]: item_context(item) for item in items}
    if len(items) != len(pool):
        raise ValueError("Duplicate pool item IDs")
    fields, users, personas = schema_fields(schema_path), [], {}
    pool_hash = hashlib.sha256(items_path.read_bytes()).hexdigest()
    for persona in read_jsonl(personas_path):
        identity = persona.get("persona_id")
        if not isinstance(identity, str) or not identity.strip() or identity in personas:
            raise ValueError("Missing or duplicate persona ID")
        personas[identity] = persona
    records = [(persona, None) for persona in personas.values()]
    if audits_path is not None:
        records, identities = [], set()
        for audit in read_jsonl(audits_path):
            identity = audit.get('persona_id')
            if not isinstance(identity, str) or not identity.strip() or identity in identities:
                raise ValueError('Missing or duplicate audit persona ID')
            identities.add(identity)
            if audit.get('status') != 'accepted':
                continue
            if identity not in personas:
                raise ValueError(f'Audit persona ID missing from personas: {identity}')
            records.append((personas[identity], audit))
    unfinished_ids = set(has_not_finished_user_id_list or [])
    if unfinished_ids:
        records = [(persona, audit) for persona, audit in records if persona['persona_id'] in unfinished_ids]
    for persona, audit in records[:limit]:
        identity = persona['persona_id']
        if audit is None:
            ids, target = persona['candidates'], persona['ground_truth']
            meta = persona['recommendation_label']
            if meta['task'] != task or meta['pool_sha256'] != pool_hash:
                raise ValueError('Label pool/task differs from supplied scene pool')
            if target != persona['label'] or not set(ids) <= set(meta['sampled_item_ids']):
                raise ValueError('Invalid target/candidate provenance')
        else:
            ids, target = audit['candidates'], audit['target']
            provenance = audit.get('simulation', {}).get('configuration', {})
            if provenance.get('pool_sha256', pool_hash) != pool_hash:
                raise ValueError('Audit label pool differs from supplied scene pool')

            meta = {'selection_status': 'accepted', 'tie_count': 0,
                    'insufficient_evidence_count': 0}
        if len(ids) != cfg["candidate_count"] or len(ids) != len(set(ids)) or any(i not in pool for i in ids):
            raise ValueError("Invalid candidate set")
        if target not in ids:
            raise ValueError('Invalid target/candidate provenance')
        reference_context = build_persona_context(persona, task, fields, cfg["evidence_chars"])

        reference = {key: reference_context[key] for key in ("interest_dimensions", "dimension_descriptions", "grounding")}
        excluded = {'topic_film', 'fam_film_studies'} if task == 'movie' else {
            'topic_music', 'fam_music_theory', 'fam_music_production', 'ind_music', 'skill_audio_production'}
        for key in reference:
            reference[key] = {k: v for k, v in reference[key].items() if k not in excluded}

        reference_atoms(reference)
        users.append({"persona_id": identity, "catalog": [pool[i] for i in ids], "target": target,
                      "private_context": build_native_persona_context(persona), "reference": reference,
                      "label_tie_count": meta["tie_count"],
                      "label_audit": label_audit(meta),
                      "label_insufficient_evidence_count": meta["insufficient_evidence_count"]})
    if not users:
        raise ValueError("No labeled users")
    return users


def prepare(output: Path, cfg: dict, users: list, expected_users: int, method: str = "iq", *, input_signature=None) -> dict:
    if len(users) != expected_users:
        raise ValueError(f"Expected {expected_users} users, found {len(users)}")
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    definition = cfg if method == "iq" else {**cfg, "method": method}
    manifest = {"configuration": definition, "signature": fingerprint(definition)}
    if input_signature is not None:
        manifest['inputs'] = input_signature
        manifest['signature'] = fingerprint({'configuration': definition, 'inputs': input_signature})
    path = output / "manifest.json"
    runtime_path = output / 'runtime.json'
    if runtime_path.exists() and json.loads(runtime_path.read_text()).get('evaluation_protocol') != EVALUATION_PROTOCOL:
        raise ValueError('Preference evaluation protocol changed; preserve old results and choose a new output directory')
    if path.exists() and json.loads(path.read_text()) != manifest:
        if (output / "runtime.json").exists() or any((output / "calls").glob("*.json")) or any((output / "users").glob("*.json")):
            raise ValueError("Run configuration changed after inference; choose a new output directory")
    write_json_atomic(path, manifest)
    rounds = cfg["Tmax"]

    attempts = rounds
    enabled = cfg.get('enable_irrationality', True)
    belief_calls = 3 if enabled else 2
    distortion_calls = cfg['M']*cfg['L'] if enabled else cfg['M']
    method_calls = (rounds+1)*belief_calls + attempts*(1 + cfg['M'] + distortion_calls)
    judge_calls = 1
    if method == "direct_llm":
        method_calls = attempts + (2 if cfg.get('final_preference_source', 'method') == 'history'
                                   else belief_calls + 1)
    elif cfg.get('final_preference_source', 'method') == 'history':
        method_calls -= belief_calls - 1
    audit = {"status": "preflight_passed", "method": method, "users": len(users), "candidate_count": cfg["candidate_count"],
        "max_method_and_ranking_calls_per_user": None,
        "max_simulator_calls_per_user": None,
        "max_judge_calls_per_user": judge_calls,
        "max_total_calls_before_retries": None,
        "calls_per_user_without_skips": {"method_and_ranking": method_calls,
                                         "simulator": attempts, "judge": judge_calls},
        "call_budget_note": "Tmax limits informative answers; answer=2 replacements add calls without a fixed bound.",
        "question_protocol": question_protocol(cfg),
        "request_protocol": request_protocol(cfg),
        **experiment_metadata(cfg),
        "simulator_protocol": SIMULATOR_PROTOCOL,
        "evaluation_policy": "terminal_only",
        "offline_checks": ["unique users", "candidate IDs and counts", "target membership",
                           "label pool hash when recorded", "active native persona rendering"],
        "results": "No empirical results exist until execution completes."}
    write_json_atomic(output / "preflight.json", audit)
    return manifest


def report_round(snapshot: dict, round_number: int) -> dict:
    """Expose actual question/answer attempts, including answer=2 replacements."""
    candidates = snapshot.get('candidates', [])
    selected_index = snapshot.get('selected_index')
    selected = candidates[selected_index]['question'] if selected_index is not None else None
    attempts = list(snapshot.get('rejected_candidates', []))
    response = snapshot.get('observed_response')
    if response is not None and (snapshot['action'] == 'ask' or 'rejected_candidates' not in snapshot):
        attempts.append({'candidate_index': selected_index, 'observed_response': response})
    presented = []
    for attempt in attempts:
        question = candidates[attempt['candidate_index']]['question']
        answer = attempt['observed_response']['answer']
        presented.append({'candidate_index': attempt['candidate_index'], 'question': question,
                          'actual_answer': answer,
                          'answer_text': question['options'].get(str(answer), 'Insufficient evidence')})
    return {**snapshot, 'round': round_number, 'selected_question': selected,
            'actual_answer': response['answer'] if response is not None else None,
            'presented_questions': presented}


def write_reports(output: Path, records: list[dict], cfg: dict, expected_users: int) -> None:
    complete = len(records) == expected_users
    terminal = []
    for record in records:
        final = record["rounds"][-1]
        evaluation = final['preference_evaluation']
        terminal.append({"persona_id": record["persona_id"], "T": record["T"],
            "interaction_cost": record["interaction_cost"], "stop_reason": record["stop_reason"],
            "questions_presented": record.get("questions_presented", record["T"]),
            "skipped_question_count": record.get("skipped_question_count", 0),
            "label_tie_count": record["label_tie_count"],
            "label_insufficient_evidence_count": record["label_insufficient_evidence_count"],
            **{k: record.get(k) for k in TIMING_METRICS},
            **final["recommendation_metrics"], **{k: evaluation[k] for k in
                ("preference_elicitation_accuracy", "inferred_atoms")}})
    metrics = RECOMMENDATION_METRICS + ("preference_elicitation_accuracy",)
    keys = metrics + ("T", "interaction_cost", "skipped_question_count") + QUESTION_METRICS
    summary = {"status": "complete" if complete else "partial", "completed_users": len(records),
        'preference_protocol': prompts.PREFERENCE_PROTOCOL,
        'request_protocol': request_protocol(cfg),
        'evaluation_protocol': EVALUATION_PROTOCOL, 'ranking_protocol': ranking_protocol(cfg),
        **experiment_metadata(cfg),
        "method": records[0].get("method", "iq") if records else None,
        "expected_users": expected_users, "candidate_count": cfg["candidate_count"],
        "evaluation_policy": "terminal_only",
        "terminal": {k: mean_present(row[k] for row in terminal) for k in keys} if terminal else {},
        'preference_score_samples': sum(row['preference_elicitation_accuracy'] is not None for row in terminal),
        'preference_score_unresolved_users': sum(row['preference_elicitation_accuracy'] is None for row in terminal),
        "scope": "Sampled synthetic personas and saved reference labels; 20 candidates; single reference target",
        "uncertainty": "No confidence intervals or baseline comparisons computed"}
    summary['label_audit'] = {
        'users_with_random_final_tie': sum(r.get('label_audit', {}).get('final_tie_resolved_randomly', False) for r in records),
        'tie_counts': [r['label_tie_count'] for r in records],
        'policy': 'Keep all users in primary results; report final-tie sensitivity separately. Labels are not regenerated.'}
    summary["timing_samples"] = {}
    for key in TIMING_METRICS:
        values = [row[key] for row in terminal if row[key] is not None]
        summary["timing_samples"][key] = len(values)
        if values:
            summary["terminal"][key] = mean(values)
    summary["timing_definition"] = (
        "elapsed_seconds: dialogue and terminal recommendation wall time, excluding offline judging; "
        "evaluation_seconds: terminal preference judge wall time; total_elapsed_seconds: both. "
        "Includes cache lookup and checkpoint overhead, excludes model discovery and report export. "
        "New session checkpoints accumulate active execution time across attempts; downtime is excluded.")
    users = []
    for record, row in zip(records, terminal):
        catalog = record.get('candidate_items', [])
        target = record.get('target')
        users.append({'persona_id': record['persona_id'],
            'rounds': [report_round(snapshot, i) for i, snapshot in enumerate(record['rounds'], 1)],
            'final_recommendation': {
                'candidates': catalog, 'ranking': record['ranking'], 'target': target,
                'target_item': next((item for item in catalog if item['item_id'] == target), None)},
            'metrics': {key: value for key, value in row.items() if key != 'persona_id'}})
    overall = {**summary['terminal'],
        'total_interaction_cost': sum(record['interaction_cost'] for record in records),
        'total_answered_questions': sum(record['T'] for record in records)}
    write_json_atomic(output / 'results.json', {
        'schema_version': 4, 'status': summary['status'], 'configuration': cfg,
        'completed_users': len(records), 'expected_users': expected_users,
        'metric_definitions': {
            'hr_at_5': '1 if the reference target is in the first 5 ranked items, otherwise 0.',
            'hr_at_10': '1 if the reference target is in the first 10 ranked items, otherwise 0.',
            'ndcg_at_5': 'Single-target discounted gain within the first 5 ranked items.',
            'ndcg_at_10': 'Single-target discounted gain within the first 10 ranked items.',
            'preference_elicitation_accuracy': 'v12 mean over inferred atom scores. Each atom is the normalized relevance-weighted mean of its reference level-pair scores; no matches scores zero.',
            'interaction_cost': 'c * T; T counts informative 0/1 answers. This is not API billing.',
            'actual_answer': '0 or 1 selects the corresponding option; 2 means insufficient evidence; null means no answer.',
            'overall': 'Means across completed users, except fields explicitly prefixed total_. Unfinished users are excluded.'},
        'overall': overall, 'users': users, 'evaluation_summary': summary})
    write_json_atomic(output / "evaluation_metrics.json", summary)
    if terminal:
        with (output / "per_user_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(terminal[0]))
            writer.writeheader()
            writer.writerows(terminal)
    old_turns = output / "per_turn_metrics.csv"
    if old_turns.exists():
        archive = output / "legacy_per_turn"
        archive.mkdir(exist_ok=True)
        old_turns.replace(archive / f"per_turn_metrics_{fingerprint(old_turns.read_text())}.csv")
    tex = ["% Generated terminal-only evaluation; never populated by dry-run fixtures.",
           f"% Status: {summary['status']}; N={len(records)}/{expected_users}; candidates={cfg['candidate_count']}",
           f"\\noindent Evaluation status: {summary['status']}; completed users: {len(records)}/{expected_users}; "
           f"candidate items per user: {cfg['candidate_count']}.\\par",
           "\\begin{tabular}{rrrrr}", "HR@5 & HR@10 & NDCG@5 & NDCG@10 & Preference alignment \\\\", "\\hline"]
    if terminal:
        tex.append(" & ".join(format_metric(summary['terminal'][k]) for k in metrics)+" \\\\")
    tex.append("\\end{tabular}")
    if terminal:
        tex.append(f"\\par Mean questions: {summary['terminal']['T']:.4f}; "
                   f"mean interaction cost: {summary['terminal']['interaction_cost']:.4f}.\\par")
    (output / "evaluation_metrics.tex").write_text("\n".join(tex)+"\n")
    report = ["# Evaluation Metrics", "", f"Status: {summary['status']}. Completed users: {len(records)}/{expected_users}.",
        f"Candidate set: {cfg['candidate_count']} saved items per user.",
        "", "Evaluation occurs only after termination; intermediate rounds retain dialogue and decision traces.",
        "", "Terminal outcomes: "+json.dumps(summary["terminal"], sort_keys=True), "",
        f"Preference score availability: {summary['preference_score_samples']}/{len(records)} users.",
        "HR@5/10 and NDCG@5/10 use the single reference target. Code splits the selected preference profile into labeled atoms;",
        "the judge finds all relevant references and assigns relevance weights for each inferred atom. "
        "Code computes sum(relevance * level_pair_score) / sum(relevance) per atom, "
        "using zero for no matches, then averages inferred atom scores.",
        ("Ranking and preference evaluation use the same summary of observed Q/A. "
         if cfg.get('final_preference_source', 'method') == 'history' else
         "Ranking and preference evaluation use the same highest-probability hypothesis. ")
        + "Scores are not posterior weighted. The selected profile, fixed atoms, reference and scores are saved for review.",
        "Accuracy is computed per user before averaging. T counts informative 0/1 answers; cost is c*T. "
        "Answer=2 removes a question from the current pool; remaining candidates are tried in descending EIG order. A round is skipped only when all candidates return 2. questions_presented includes all attempts; skipped_question_count counts answer=2 questions.",
        "Empty checklists score zero. Reference labels may be ambiguous; consult the labeling provenance.",
        "No human-user outcomes, confidence intervals or baseline superiority claims are implied.", "",
        "Combined report: results.json (rounds, answers, item candidates, targets, per-user and overall metrics).",
        "Additional exports: per_user_metrics.csv, evaluation_metrics.json and users/*.json."]
    (output / "evaluation_metrics.md").write_text("\n".join(report)+"\n")


def execute(output: Path, cfg: dict, users: list, manifest: dict, max_users: int | None,
            method: str = "iq", client_factory=None, client_factories=None) -> None:
    if max_users is not None and max_users < 1:
        raise ValueError("max-users must be positive")
    clients, digests = {}, {}
    for role in ("recommender", "simulator", "judge"):
        factory = (client_factories or {}).get(role) or client_factory or OllamaClient
        client = factory(cfg[f"{role}_model"], cfg["host"], timeout=cfg["timeout"],
            num_ctx=cfg["num_ctx"], num_predict=cfg["num_predict"], retries=cfg["retries"])
        clients[role] = client
        digests[role] = client.check_model()
    runtime = {"model_digests": digests, "temperature": 0, "manifest_signature": manifest["signature"],
               'preference_protocol': prompts.PREFERENCE_PROTOCOL,
               'request_protocol': request_protocol(cfg),
               "question_protocol": question_protocol(cfg), 'evaluation_protocol': EVALUATION_PROTOCOL,
               'ranking_protocol': ranking_protocol(cfg), 'simulator_protocol': SIMULATOR_PROTOCOL,
               **experiment_metadata(cfg)}
    api_backends = {role: client.runtime_metadata for role, client in clients.items()
                    if hasattr(client, 'runtime_metadata')}
    if api_backends:
        runtime.update(api_backends=api_backends, temperature=None)
    if method == "iq":
        if cfg.get('enable_irrationality', True):
            runtime["scoring_protocol"] = "distortion_per_pair_vectors_v3"
            runtime["behavior_id_policy"] = "code_assigned_v1"
        else:
            runtime['scoring_protocol'] = 'distortion_per_preference_vectors_v1'
    elif method == "direct_llm":
        runtime["terminal_protocol"] = "direct_llm_eig_terminal_v1"
    path = output / "runtime.json"
    if path.exists() and json.loads(path.read_text()) != runtime:
        previous = json.loads(path.read_text())
        if previous.get('simulator_protocol') != SIMULATOR_PROTOCOL:
            raise ValueError('Simulator persona protocol changed; preserve old checkpoints and choose a new output directory')
        if previous.get("question_protocol") != runtime["question_protocol"]:
            raise ValueError(f"Question protocol changed from {previous.get('question_protocol')} "
                             f"to {runtime['question_protocol']}; preserve old checkpoints and choose a new output directory")
        if previous.get('evaluation_protocol') != EVALUATION_PROTOCOL:
            raise ValueError('Preference evaluation protocol changed; preserve old results and choose a new output directory')
        if previous.get('preference_protocol') != prompts.PREFERENCE_PROTOCOL:
            raise ValueError('Preference description format changed; preserve old results and choose a new output directory')
        if previous.get('request_protocol') != runtime['request_protocol']:
            raise ValueError('Model request batching or output schema changed; preserve old results and choose a new output directory')
        raise ValueError("Model digest or runtime changed; choose a new output directory")
    write_json_atomic(path, runtime)
    calls = Calls(output / "calls", cfg, clients, fingerprint(runtime))
    agent, records = METHODS[method](cfg, calls), []
    for user in users[:max_users]:
        path = output / "users" / f"{fingerprint(user['persona_id'])}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["runtime_signature"] != fingerprint(runtime) or saved["record_hash"] != fingerprint(saved["record"]):
                raise ValueError("Corrupt user checkpoint")
            record = copy.deepcopy(saved["record"])
            reject_schema_conditioned_trace(record)
        else:
            user_started = time.monotonic()
            evaluation_seconds = 0.0
            checkpoint_path = output / "session_checkpoints" / f"{fingerprint(user['persona_id'])}.json"
            saved_state = None
            timing = {"total_elapsed_seconds": 0.0, "evaluation_seconds": 0.0, "execution_attempts": 0,
                      "started_at": datetime.now(timezone.utc).isoformat()}
            if checkpoint_path.exists():
                saved_session = json.loads(checkpoint_path.read_text())
                if (saved_session["runtime_signature"] != fingerprint(runtime)
                        or saved_session["persona_id"] != user["persona_id"]
                        or saved_session["checkpoint_hash"] != fingerprint(saved_session["checkpoint"])):
                    raise ValueError("Corrupt session checkpoint")
                saved_state = saved_session["checkpoint"]["state"]
                timing = saved_session["checkpoint"]["timing"]
            latest_state = saved_state
            def save_session(state):
                nonlocal latest_state
                latest_state = state
                payload = {"state": state, "timing": {
                    "total_elapsed_seconds": timing["total_elapsed_seconds"] + time.monotonic()-user_started,
                    "evaluation_seconds": timing["evaluation_seconds"] + evaluation_seconds,
                    "execution_attempts": timing["execution_attempts"]+1, "started_at": timing["started_at"]}}
                write_json_atomic(checkpoint_path, {"runtime_signature": fingerprint(runtime),
                    "persona_id": user["persona_id"], "checkpoint": payload, "checkpoint_hash": fingerprint(payload)})
            write_json_atomic(output / "progress.json", {"status": "running", "method": method,
                "persona_id": user["persona_id"], "answered_questions": len(saved_state["history"]) if saved_state else 0,
                "skipped_question_count": len(uninformative_questions(saved_state["rounds"])) if saved_state else 0,
                "completed_users": len(records), "expected_users": len(users)})
            def answer(history, question):
                return simulate(calls, user["private_context"], history, question, user["catalog"])
            def on_round(snapshot):
                nonlocal evaluation_seconds
                if snapshot["action"] == "stop":
                    inferred = snapshot.get('final_preference', snapshot["belief"])
                    snapshot["recommendation_metrics"] = recommendation_metrics(snapshot["ranking"], user["target"])
                    evaluation_started = time.monotonic()
                    try:
                        snapshot["preference_evaluation"] = evaluate_preference(calls, user["reference"], inferred)
                    finally:
                        evaluation_seconds += time.monotonic() - evaluation_started
                turn_file = f"{snapshot['answered_questions']:02d}"
                if snapshot["action"] == "skip":
                    turn_file += f"_skip_{snapshot['questions_presented']:04d}"
                write_json_atomic(output / "rounds" / fingerprint(user["persona_id"]) /
                                  f"{turn_file}.json", snapshot)
                effective = snapshot["answered_questions"] + int(snapshot["action"] == "ask")
                write_json_atomic(output / "progress.json", {"status": "running", "persona_id": user["persona_id"],
                    "answered_questions": effective, "completed_users": len(records),
                    "questions_presented": snapshot["questions_presented"],
                    "skipped_question_count": snapshot["questions_presented"] - effective,
                    "expected_users": len(users)})
                print(f"{user['persona_id']}: answered={snapshot['answered_questions']}, action={snapshot['action']}", flush=True)
            try:
                record = session(agent, user["catalog"], answer, on_round,
                                 resume_state=saved_state, on_checkpoint=save_session)
            except Exception as exc:
                if latest_state is not None:
                    save_session(latest_state)
                progress = json.loads((output / "progress.json").read_text())
                write_json_atomic(output / "progress.json", {**progress, "status": "failed", "error": str(exc)})
                raise
            total_elapsed = timing["total_elapsed_seconds"] + time.monotonic() - user_started
            total_evaluation = timing["evaluation_seconds"] + evaluation_seconds
            record.update(persona_id=user["persona_id"], label_tie_count=user["label_tie_count"],
                          label_audit=user.get('label_audit', {}),
                          label_insufficient_evidence_count=user["label_insufficient_evidence_count"],
                          started_at=timing["started_at"], finished_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=total_elapsed-total_evaluation,
                          evaluation_seconds=total_evaluation, total_elapsed_seconds=total_elapsed,
                          execution_attempts=timing["execution_attempts"]+1)

        record.update(candidate_items=user['catalog'], target=user['target'])
        write_json_atomic(path, {"runtime_signature": fingerprint(runtime), "record": record,
                                 "record_hash": fingerprint(record)})
        records.append(record)
        write_reports(output, records, cfg, len(users))
    write_json_atomic(output / "progress.json", {"status": "complete" if len(records) == len(users) else "partial",
                      "completed_users": len(records), "expected_users": len(users)})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['movie', 'music'], default='movie')
    parser.add_argument("--config", type=Path, default=ROOT / "IQ_recommender_config.json")
    parser.add_argument("--personas", type=Path, default=ROOT / "sampled_personas.jsonl", help="Active personas (default: sampled_personas.jsonl)")
    parser.add_argument("--audits", type=Path, help="Targets and candidates (default: <task>_persona_audits.jsonl)")
    parser.add_argument("--items", type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--method", choices=[*METHODS, "pebol", "decisive", "gate"], default="iq")
    parser.add_argument('--eig-stop-policy', choices=['cost', 'budget'],
                        help='cost: stop when max EIG <= c (default); budget: continue until Tmax informative answers')
    parser.add_argument('--enable-irrationality', '--enable_irrationality',
                        action=argparse.BooleanOptionalAction, default=None,
                        help='Include irrationality hypotheses (default); --no-enable-irrationality uses only preferences')
    parser.add_argument('--final-preference-source', choices=['method', 'history'],
                        help='method: original terminal profile (default); history: shared direct summary for ranking and judging')
    parser.add_argument('--backbone', help='deepseek-thinking-low, deepseek-thinking-disabled, claude-sonnet-low, or an Ollama model name',default='deepseek-thinking-low')
    parser.add_argument('--ollama-host', default='http://localhost:11434',
                        help='Ollama endpoint for a model selected with --backbone')
    parser.add_argument("--pebol-config", type=Path, help="PEBOL-specific configuration for --method pebol")
    parser.add_argument("--decisive-config", type=Path, help="DECISIVE-specific configuration for --method decisive")
    parser.add_argument("--gate-config", type=Path, help="GATE-specific configuration for --method gate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-users", type=int, default=200,
                        help="Sample this many accepted users when the unfinished-ID list is empty (default: 20)")
    parser.add_argument('--has-not-finished-user-id-list', '--has_not_finished_user_id_list', nargs='*',
                        default=[],
                        help='Only select these persona_id strings; an empty list uses normal sampling')
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs without contacting any backend")
    parser.add_argument("--max-users", type=int, default=200,help="Limit execution for debugging; report stays partial")
    args = parser.parse_args(argv)
    if args.expected_users < 1:
        parser.error('--expected-users must be positive')

    args.audits = args.audits or ROOT / f'{args.task}_persona_audits.jsonl'
    pool_name = 'movies' if args.task == 'movie' else 'musical'
    args.items = args.items or ROOT / 'task_dataset' / f'{pool_name}_sample_1000.jsonl'
    if args.gate_config is not None and args.method != "gate":
        parser.error('--gate-config requires --method gate')
    if args.method == "gate":
        if any(value is not None for value in (
                args.eig_stop_policy, args.final_preference_source, args.enable_irrationality,
                args.pebol_config, args.decisive_config)):
            parser.error('IQ ablation flags and other baseline configurations do not apply to GATE')
        if __package__:
            from .baseline.GATE import run as run_gate
        else:
            import sys
            sys.path.insert(0, str(ROOT.parent))
            from rec_simulator.baseline.GATE import run as run_gate
        forwarded = ["--prepare-only" if args.prepare_only else "--execute"]
        for key in ("config", "personas", "audits", "items", "schema", "expected_users",
                    "output", "max_users", "gate_config", "backbone", "ollama_host", "task"):
            value = getattr(args, key)
            if value is not None:
                forwarded.extend(["--" + key.replace("_", "-"), str(value)])
        if args.has_not_finished_user_id_list:
            forwarded.extend(["--has-not-finished-user-id-list", *args.has_not_finished_user_id_list])
        return run_gate(forwarded)
    if args.decisive_config is not None and args.method != "decisive":
        parser.error('--decisive-config requires --method decisive')
    if args.method == "decisive":
        if any(value is not None for value in (args.eig_stop_policy, args.final_preference_source,
                                              args.enable_irrationality, args.pebol_config)):
            parser.error('IQ ablation flags and --pebol-config do not apply to DECISIVE')
        if __package__:
            from .baseline.DECISIVE import run as run_decisive
        else:
            import sys
            sys.path.insert(0, str(ROOT.parent))
            from rec_simulator.baseline.DECISIVE import run as run_decisive
        forwarded = ["--prepare-only" if args.prepare_only else "--execute"]
        for key in ("config", "personas", "audits", "items", "schema", "expected_users",
                    "output", "max_users", "decisive_config", "backbone", "ollama_host", "task"):
            value = getattr(args, key)
            if value is not None:
                forwarded.extend(["--" + key.replace("_", "-"), str(value)])
        if args.has_not_finished_user_id_list:
            forwarded.extend(["--has-not-finished-user-id-list", *args.has_not_finished_user_id_list])
        return run_decisive(forwarded)
    if args.method == "pebol":
        if any(value is not None for value in (args.eig_stop_policy, args.final_preference_source,
                                              args.enable_irrationality)):
            parser.error('These ablation flags apply only to iq and direct_llm')

        if __package__:
            from .baseline.PEBOL import run as run_pebol
        else:
            import sys
            sys.path.insert(0, str(ROOT.parent))
            from rec_simulator.baseline.PEBOL import run as run_pebol
        forwarded = ["--prepare-only" if args.prepare_only else "--execute"]
        for key in ("config", "personas", "audits", "items", "schema", "expected_users", "output",
                    "max_users", "pebol_config", "backbone", "ollama_host", "task"):
            value = getattr(args, key)
            if value is not None:
                forwarded.extend(["--" + key.replace("_", "-"), str(value)])
        if args.has_not_finished_user_id_list:
            forwarded.extend(['--has-not-finished-user-id-list', *args.has_not_finished_user_id_list])
        return run_pebol(forwarded)
    args.personas = args.personas or ROOT / 'sampled_personas.jsonl'
    cfg = json.loads(args.config.read_text())
    cfg['task'] = args.task
    for key in ('eig_stop_policy', 'final_preference_source', 'enable_irrationality'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    backbone = resolve_backbone(args.backbone, args.ollama_host) if args.backbone is not None else None
    if backbone is not None:
        cfg.update({f'{role}_model': backbone['model'] for role in ('recommender', 'simulator', 'judge')})
        cfg['host'] = backbone['host']

    if args.output is None:
        backbone_name = backbone['name'] if backbone is not None else cfg['recommender_model']
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', backbone_name).strip('._-') or 'model'
        ablation = '' if cfg.get('enable_irrationality', True) else '_no_irrationality'
        name = f"{safe_name}_{args.method}_{args.task}{ablation}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        args.output = ROOT / 'recommender_runs' / name
    if args.max_users is not None and args.max_users < 1:
        parser.error("--max-users must be positive")

    resolved_output = args.output.resolve()
    if any(p.resolve() == resolved_output or resolved_output in p.resolve().parents
           for p in (args.config, args.personas, args.audits, args.items, args.schema)):
        raise ValueError("Output directory contains an input")
    users = load_dataset(args.personas, args.items, args.schema, cfg,
                         audits_path=args.audits,
                         limit=None if args.has_not_finished_user_id_list else args.expected_users,
                         has_not_finished_user_id_list=args.has_not_finished_user_id_list)

    signature = {'persona_ids': [user['persona_id'] for user in users],
                 'dataset_sha256': fingerprint(users)}
    if backbone is not None:
        signature['backbone'] = backbone
    expected_users = len(users) if args.has_not_finished_user_id_list else args.expected_users
    manifest = prepare(args.output, cfg, users, expected_users, args.method,
                       input_signature=signature)
    print(f"Prepared {len(users)} users x {cfg['candidate_count']} candidates: {args.output}", flush=True)
    if not args.prepare_only:
        if backbone is None:
            execute(args.output, cfg, users, manifest, args.max_users, args.method)
        else:
            execute(args.output, cfg, users, manifest, args.max_users, args.method,
                    client_factory=backbone_client_factory(backbone))
    else:
        print("Offline preparation complete; no model backend was contacted.")


if __name__ == "__main__":
    main()
