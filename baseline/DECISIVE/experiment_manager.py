"""Independent DECISIVE runner: offline preflight, API execution, exact replay."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import time

from ... import llm_recommender as shared
from .. import terminal

ROOT = Path(__file__).resolve().parent


def load_config(path, decisive_path, backbone, *, task=None):
    common = json.loads(path.read_text())
    task = task if task is not None else common.get("task", "movie")
    if task not in {"movie", "music"}:
        raise ValueError("DECISIVE supports movie and music tasks")
    common.update(task=task, host=backbone["host"],
                  **{f"{role}_model": backbone["model"] for role in ("recommender", "simulator", "judge")})
    shared.validate_config(common)
    terminal.terminal_metadata(common)
    config = json.loads(decisive_path.read_text())
    if set(config) != set(json.loads((ROOT / "config.json").read_text())):
        raise ValueError("DECISIVE configuration keys differ from config.json")
    for key in ("particles", "max_factors"):
        if type(config[key]) is not int or config[key] < 2:
            raise ValueError(f"{key} must be an integer >= 2")
    for key in ("dirichlet_alpha", "kappa", "confidence_threshold", "min_eig_bits"):
        if not shared.number(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if config["confidence_threshold"] > 1:
        raise ValueError("confidence_threshold must be <= 1")
    return common, config


def prepare(output, common, config, users, inputs, backbone):
    sources = sorted([*ROOT.glob("*.py"), *shared.ROOT.glob("*.py"), ROOT / "PAPER.json"])
    sources.append(Path(terminal.__file__))
    definition = {
        "method": "decisive", "common_configuration": common, "decisive_configuration": config,
        "backbone": backbone, "persona_ids": [u["persona_id"] for u in users],
        "dataset_signature": shared.fingerprint(users),
        "inputs_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items()},
        "sources_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "evaluation_protocol": shared.EVALUATION_PROTOCOL,
        "ranking_protocol": shared.ranking_protocol(common), **terminal.terminal_metadata(common)}
    manifest = {"configuration": definition, "signature": shared.fingerprint(definition)}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        if (output / "runtime.json").exists() or any((output / "calls").glob("*.json")):
            raise ValueError("Inputs, config or source changed; choose a new output directory")
    shared.write_json_atomic(path, manifest)
    shared.write_json_atomic(output / "preflight.json", {
        "status": "preflight_passed", "method": "decisive", "users": len(users),
        "persona_ids": definition["persona_ids"], "candidate_count": common["candidate_count"],
        "Tmax": common["Tmax"], "backbone": backbone, "decisive_configuration": config,
        "grounding_calls_per_catalog_before_retries": 1 + common["candidate_count"],
        "max_presented_questions_per_user": config["max_factors"] * (config["max_factors"] - 1) // 2,
        "max_informative_answers": common["Tmax"], "max_terminal_inference_calls": terminal.terminal_call_budget(common), "max_judge_calls": 1,
        "results": "Offline preparation only; no clients, model requests or empirical results."})
    return manifest


def execute(output, common, config, users, manifest, max_users=None, *, client_factory):
    from .grounding import Calls, build_grounding
    from .dialogue_sim import run_dialogue
    from .eval_manager import write_reports

    clients = {role: client_factory(common[f"{role}_model"], common["host"],
               timeout=common["timeout"], num_ctx=common["num_ctx"],
               num_predict=common["num_predict"], retries=common["retries"])
               for role in ("recommender", "simulator", "judge")}
    packages = {}
    for name in ("numpy", "openai", "anthropic", "python-dotenv", "jsonschema"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    runtime = {"manifest_signature": manifest["signature"], "package_versions": packages,
               "model_digests": {role: client.check_model() for role, client in clients.items()},
               "api_backends": {role: client.runtime_metadata for role, client in clients.items()
                                if hasattr(client, "runtime_metadata")}}
    path = output / "runtime.json"
    if path.exists() and json.loads(path.read_text()) != runtime:
        raise ValueError("Runtime or model revision changed; choose a new output directory")
    shared.write_json_atomic(path, runtime)
    signature = shared.fingerprint(runtime)
    calls = Calls(output / "calls", common, clients, signature)
    records = []
    for user in users[:max_users]:
        identity = user["persona_id"]
        key = shared.fingerprint(identity)
        result_path = output / "users" / f"{key}.json"
        checkpoint_path = output / "session_checkpoints" / f"{key}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if (saved["runtime_signature"] != signature
                    or saved["record_hash"] != shared.fingerprint(saved["record"])):
                raise ValueError("Corrupt DECISIVE user result")
            record = saved["record"]
        else:
            previous, elapsed, evaluation, attempts = [], 0., 0., 0
            started_at = datetime.now(timezone.utc).isoformat()
            if checkpoint_path.exists():
                saved = json.loads(checkpoint_path.read_text())
                if (saved["runtime_signature"] != signature or saved["persona_id"] != identity
                        or saved["checkpoint_hash"] != shared.fingerprint(saved["checkpoint"])):
                    raise ValueError("Corrupt DECISIVE checkpoint")
                state = saved["checkpoint"]
                previous = state["rounds"]
                elapsed, evaluation, attempts = state["elapsed"], state["evaluation"], state["attempts"]
                started_at = state["started_at"]
            started = time.monotonic()
            latest, current_evaluation = previous, 0.

            def checkpoint(rounds):
                nonlocal latest
                latest = copy.deepcopy(rounds)
                state = {"rounds": latest, "elapsed": elapsed + time.monotonic() - started,
                         "evaluation": evaluation + current_evaluation, "attempts": attempts + 1,
                         "started_at": started_at}
                shared.write_json_atomic(checkpoint_path, {
                    "persona_id": identity, "runtime_signature": signature,
                    "checkpoint": state, "checkpoint_hash": shared.fingerprint(state)})
                if rounds:
                    shared.write_json_atomic(output / "rounds" / key / f"{len(rounds):03d}.json", rounds[-1])
                shared.write_json_atomic(output / "progress.json", {
                    "status": "running", "method": "decisive", "persona_id": identity,
                    "answered_questions": sum(r["action"] == "ask" for r in rounds),
                    "questions_presented": sum(r["action"] != "stop" for r in rounds),
                    "completed_users": len(records), "expected_users": len(users)})

            try:
                checkpoint(previous)
                grounding = build_grounding(user["catalog"], calls, config)
                shared.write_json_atomic(output / "grounding" / f"{grounding['catalog_signature']}.json", grounding)
                seed = shared.stable_seed(common["seed"], identity, grounding["catalog_signature"])
                record = run_dialogue(grounding, common, config, seed, calls,
                    lambda history, question: shared.simulate(
                        calls, user["private_context"], history, question, user["catalog"]),
                    saved_rounds=previous, on_checkpoint=checkpoint, catalog=user["catalog"])
                final = record["rounds"][-1]
                final["recommendation_metrics"] = shared.recommendation_metrics(record["ranking"], user["target"])
                judge_started = time.monotonic()
                try:
                    final["preference_evaluation"] = shared.evaluate_preference(
                        calls, user["reference"], final["belief"])
                finally:
                    current_evaluation += time.monotonic() - judge_started
                shared.write_json_atomic(output / "rounds" / key / f"{len(record['rounds']):03d}.json", final)
            except Exception as exc:
                checkpoint(latest)
                shared.write_json_atomic(output / "progress.json", {
                    "status": "failed", "method": "decisive", "persona_id": identity,
                    "error": str(exc), "completed_users": len(records), "expected_users": len(users)})
                raise
            total, judge_total = elapsed + time.monotonic() - started, evaluation + current_evaluation
            record.update(persona_id=identity, candidate_items=user["catalog"], target=user["target"],
                          label_tie_count=user["label_tie_count"],
                          label_insufficient_evidence_count=user["label_insufficient_evidence_count"],
                          label_audit=user.get("label_audit", {}),
                          started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=total - judge_total, evaluation_seconds=judge_total,
                          total_elapsed_seconds=total, execution_attempts=attempts + 1)
            shared.write_json_atomic(result_path, {"runtime_signature": signature, "record": record,
                                                   "record_hash": shared.fingerprint(record)})
        records.append(record)
        write_reports(output, records, common, config, len(users))
        print(f"DECISIVE {identity}: T={record['T']}; stop={record['stop_reason']}; "
              f"completed={len(records)}/{len(users)}", flush=True)
    shared.write_json_atomic(output / "progress.json", {
        "status": "complete" if len(records) == len(users) else "partial", "method": "decisive",
        "completed_users": len(records), "expected_users": len(users)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=shared.ROOT / "llm_recommender_config.json")
    parser.add_argument("--decisive-config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--task", choices=["movie", "music"], default="movie")
    parser.add_argument("--backbone", default="deepseek-thinking-low")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--personas", type=Path, default=shared.ROOT / "sampled_personas.jsonl")
    parser.add_argument("--audits", type=Path)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--schema", type=Path, default=shared.DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-users", type=int, default=20)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--has-not-finished-user-id-list", "--has_not_finished_user_id_list",
                        nargs="*", default=[])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_users < 1 or (args.max_users is not None and args.max_users < 1):
        parser.error("User limits must be positive")
    backbone = shared.resolve_backbone(args.backbone, args.ollama_host)
    common, config = load_config(args.config, args.decisive_config, backbone, task=args.task)
    if args.items is None:
        pool = "movies" if args.task == "movie" else "musical"
        args.items = shared.ROOT / "task_dataset" / f"{pool}_sample_1000.jsonl"
    if args.audits is None and args.personas == shared.ROOT / "sampled_personas.jsonl":
        args.audits = shared.ROOT / f"{args.task}_persona_audits.jsonl"
    users = shared.load_dataset(args.personas, args.items, args.schema, common,
        audits_path=args.audits, limit=None if args.has_not_finished_user_id_list else args.expected_users,
        has_not_finished_user_id_list=args.has_not_finished_user_id_list)
    if args.has_not_finished_user_id_list:
        missing = set(args.has_not_finished_user_id_list) - {u["persona_id"] for u in users}
        if missing:
            raise ValueError(f"Requested user IDs have no accepted data: {sorted(missing)}")
    elif len(users) != args.expected_users:
        raise ValueError(f"Expected {args.expected_users} users, found {len(users)}")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", backbone["name"]).strip("._-") or "model"
    output = args.output or shared.ROOT / "recommender_runs" / (
        f"{safe_name}_decisive_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    inputs = {k: getattr(args, k) for k in ("config", "decisive_config", "personas", "items", "schema")}
    if args.audits is not None:
        inputs["audits"] = args.audits
    resolved = output.resolve()
    if (any(p.resolve() == resolved or resolved in p.resolve().parents for p in inputs.values())
            or resolved == ROOT or resolved in ROOT.parents
            or (ROOT in resolved.parents and ROOT / "runs" not in resolved.parents)):
        raise ValueError("Output must not contain inputs or overwrite source; use a separate run directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A DECISIVE process is already using this output directory")
        prepare_manifest = prepare(output, common, config, users, inputs, backbone)
        print(f"Prepared DECISIVE: {len(users)} users x {common['candidate_count']} candidates; "
              f"output={output}", flush=True)
        if args.execute:
            execute(output, common, config, users, prepare_manifest, args.max_users,
                    client_factory=shared.backbone_client_factory(backbone))
        else:
            print("Offline preparation complete; no model backend was contacted.")
    return 0
