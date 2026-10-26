"""Shared metric functions, DECISIVE-specific ranking and reporting metadata."""
import csv
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil

from ... import llm_recommender as shared
from .. import terminal
from .core import QUESTION_PROTOCOL, RANKING_PROTOCOL
from .diagnostics import load_records, record_diagnostics
from .dialogue_sim import LEGACY_SUMMARY_PROTOCOL
from .grounding import SCORING_PROTOCOL


def write_reports(output, records, common, config, expected_users):
    summary_protocols = sorted({record.get("summary_protocol", LEGACY_SUMMARY_PROTOCOL)
                                for record in records})
    if len(summary_protocols) > 1:
        raise ValueError("Cannot combine different DECISIVE summary protocols")
    terminal_metadata = (terminal.terminal_report_metadata(records, common)
                         if any("terminal_protocol" in record for record in records) else {})
    if terminal_metadata:
        terminal_metadata["preference_score_note"] = (
            "The shared judge evaluates the same EIG MAP preference used for ranking. "
            "Numerical utility beliefs remain acquisition diagnostics. Empty dialogues use EIG's uniform prior.")
    rows, users = [], []
    for record in records:
        final = record["rounds"][-1]
        row = {"persona_id": record["persona_id"],
               **{k: record[k] for k in (
                   "T", "interaction_cost", "stop_reason", "questions_presented",
                   "skipped_question_count", "label_tie_count", "label_insufficient_evidence_count",
                   *shared.TIMING_METRICS)},
               **final["recommendation_metrics"],
               **{k: final["preference_evaluation"][k] for k in (
                   "preference_elicitation_accuracy", "inferred_atoms")},
               "decision_confidence": final["state_before"]["confidence"]}
        rows.append(row)
        rounds = []
        for i, snapshot in enumerate(record["rounds"], 1):
            q, response = snapshot.get("question"), snapshot.get("observed_response")
            answer = response["answer"] if response is not None else None
            rounds.append({**snapshot, "round": i, "selected_question": q, "actual_answer": answer,
                           "presented_questions": ([{"question": q, "actual_answer": answer,
                               "answer_text": q["options"].get(str(answer), "Insufficient evidence")}]
                               if q is not None else [])})
        catalog, target = record["candidate_items"], record["target"]
        users.append({"persona_id": record["persona_id"], "rounds": rounds,
                      "grounding": record["grounding"], "particle_seed": record["particle_seed"],
                      "final_recommendation": {"candidates": catalog, "target": target,
                          "ranking": record["ranking"],
                          "target_item": next(item for item in catalog if item["item_id"] == target)},
                      "metrics": {k: v for k, v in row.items() if k != "persona_id"}})
    metrics = (*shared.RECOMMENDATION_METRICS, "preference_elicitation_accuracy", "T",
               "interaction_cost", "questions_presented", "skipped_question_count",
               "decision_confidence", *shared.TIMING_METRICS)
    summary = {
        "method": "decisive", "task": common.get("task", "movie"),
        "status": "complete" if len(records) == expected_users else "partial",
        "completed_users": len(records), "expected_users": expected_users,
        "candidate_count": common["candidate_count"], "question_protocol": QUESTION_PROTOCOL,
        "ranking_protocol": RANKING_PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
        "evaluation_protocol": shared.EVALUATION_PROTOCOL, "simulator_protocol": shared.SIMULATOR_PROTOCOL,
        "evaluation_policy": "terminal_only",
        "terminal": {key: shared.mean_present(row[key] for row in rows) for key in metrics},
        "preference_score_samples": sum(r["preference_elicitation_accuracy"] is not None for r in rows),
        "preference_score_unresolved_users": sum(r["preference_elicitation_accuracy"] is None for r in rows),
        "preference_score_empty_profile_users": sum(r["inferred_atoms"] == 0 for r in rows),
        "timing_samples": {key: sum(row[key] is not None for row in rows) for key in shared.TIMING_METRICS},
        "label_audit": {
            "users_with_random_final_tie": sum(
                r.get("label_audit", {}).get("final_tie_resolved_randomly", False) for r in records),
            "tie_counts": [r["label_tie_count"] for r in records],
            "policy": "Keep all saved users and tournament labels; no relabeling."},
        "reporting_protocol": "decisive_shared_exports_v2",
        "report_sources_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("diagnostics.py"))},
        "ranking_definition": "Descending posterior expected utility S @ E[w]; stable input-order ties.",
        "preference_recovery_representation": "evaluation_only_dialogue_summary",
        "summary_protocol": summary_protocols[0] if summary_protocols else None,
        "summary_prompt_sha256": records[0].get("summary_prompt_sha256") if records else None,
        "preference_score_note": "Shared judge evaluates a dialogue summary, not the numerical weight posterior. "
            "Relative factor importance does not establish absolute liking. A zero-question run has no inferred atoms.",
        "timing_definition": "Active grounding, inference, simulator, terminal preference and ranking time in elapsed_seconds; "
            "shared preference judge in evaluation_seconds. Excludes downtime, model discovery and report export.",
        "adaptation": f"Single selected backbone, public {common.get('task', 'movie')} metadata, native persona answers and saved targets. "
            "Paper-based reconstruction, not author code or original benchmark replication.",
        **terminal_metadata}
    shared.write_json_atomic(output / "results.json", {
        "schema_version": 4, "method": "decisive", "status": summary["status"],
        "configuration": common, "decisive_configuration": config,
        "completed_users": len(records), "expected_users": expected_users,
        "overall": {**summary["terminal"],
                    "total_interaction_cost": sum(r["interaction_cost"] for r in rows),
                    "total_answered_questions": sum(r["T"] for r in rows)},
        "users": users, "evaluation_summary": summary,
        "metric_definitions": {
            "hr_at_5": "1 if the reference target is in the first 5 ranked items, otherwise 0.",
            "hr_at_10": "1 if the reference target is in the first 10 ranked items, otherwise 0.",
            "ndcg_at_5": "Single-target discounted gain within the first 5 ranked items.",
            "ndcg_at_10": "Single-target discounted gain within the first 10 ranked items.",
            "preference_elicitation_accuracy": "Shared v12 atom alignment of the saved terminal preference.",
            "interaction_cost": "c * T; informative 0/1 answers only; not API billing.",
            "actual_answer": "0 selects the first factor; 1 the second; 2 skips with no posterior update; null means no answer.",
            "overall": "Means across completed users, except fields explicitly prefixed total_."}})
    shared.write_json_atomic(output / "evaluation_metrics.json", summary)
    shared.write_json_atomic(output / "diagnostics.json", {
        "method": "decisive", "scope": "Post-run diagnostics only; never fed to inference.",
        "users": [record_diagnostics(record) for record in records]})
    if rows:
        with (output / "per_user_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# DECISIVE terminal evaluation", "",
             f"Status: {summary['status']}; completed users: {len(records)}/{expected_users}.", "",
             "| Metric | Mean |", "| --- | ---: |"]
    lines += [f"| {k} | {shared.format_metric(v, 6)} |" for k, v in summary["terminal"].items()]
    lines += ["", summary["ranking_definition"], "", summary["preference_score_note"], "",
              f"Preference scores available: {summary['preference_score_samples']}/{len(rows)}; "
              f"unresolved: {summary['preference_score_unresolved_users']}; "
              f"empty absolute profiles: {summary['preference_score_empty_profile_users']}.", "",
              summary["adaptation"], "", summary["timing_definition"], ""]
    (output / "evaluation_metrics.md").write_text("\n".join(lines), encoding="utf-8")
    tex = [f"% DECISIVE terminal evaluation; status={summary['status']}; N={len(records)}/{expected_users}",
           f"\\noindent Evaluation status: {summary['status']}; completed users: "
           f"{len(records)}/{expected_users}; candidate items per user: {common['candidate_count']}.\\par",
           "\\begin{tabular}{rrrrr}",
           "HR@5 & HR@10 & NDCG@5 & NDCG@10 & Preference alignment \\\\",
           "\\hline"]
    if rows:
        tex.append(" & ".join(shared.format_metric(summary["terminal"][key]) for key in
            (*shared.RECOMMENDATION_METRICS, "preference_elicitation_accuracy")) + " \\\\")
    tex.append("\\end{tabular}")
    if rows:
        tex.append(f"\\par Mean questions: {summary['terminal']['T']:.4f}; "
                   f"mean interaction cost: {summary['terminal']['interaction_cost']:.4f}.\\par")
    (output / "evaluation_metrics.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def main(argv=None):
    """Regenerate reports from validated completed records without model calls."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.run
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A DECISIVE process is already using this output directory")
        config = json.loads((output / "manifest.json").read_text())["configuration"]
        if config["method"] != "decisive":
            raise ValueError("Expected a DECISIVE run")
        records = load_records(output)
        backup = output / "report_backups" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup.mkdir(parents=True)
        for name in ("results.json", "per_user_metrics.csv", "evaluation_metrics.json",
                     "evaluation_metrics.md", "evaluation_metrics.tex", "diagnostics.json"):
            if (output / name).exists():
                shutil.copy2(output / name, backup / name)
        write_reports(output, records, config["common_configuration"],
                      config["decisive_configuration"], len(config["persona_ids"]))
    print(f"Regenerated DECISIVE reports: {output}; previous reports: {backup}; no API calls.")


if __name__ == "__main__":
    main()
