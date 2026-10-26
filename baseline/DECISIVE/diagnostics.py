"""Offline diagnostics; saved targets are used only to evaluate saved rankings."""
import argparse
import json
from pathlib import Path

import numpy as np

from ... import llm_recommender as shared
from .core import DecisionModel


def record_diagnostics(record):
    grounding = record["grounding"]
    scores = np.asarray(grounding["matrix"], dtype=float)
    ids = grounding["item_ids"]
    final = record["rounds"][-1]
    state = final["state_before"]
    weights = np.asarray(state["particle_weights"])
    # A dominated option cannot win under strictly positive Dirichlet weights.
    dominates = (scores[:, None, :] >= scores[None, :, :] - 1e-12).all(axis=2)
    dominates &= (scores[:, None, :] > scores[None, :, :] + 1e-12).any(axis=2)
    equal = np.isclose(scores[:, None, :], scores[None, :, :], rtol=0, atol=1e-12).all(axis=2)
    groups = [[ids[j] for j in np.flatnonzero(equal[i])]
              for i in range(len(ids)) if equal[i].sum() > 1 and not equal[i, :i].any()]
    informative, skipped = [], []
    for snapshot in record["rounds"]:
        if snapshot["action"] in ("ask", "skip"):
            (informative if snapshot["action"] == "ask" else skipped).append(snapshot["factor_indices"])
    coverage = []
    for i, factor in enumerate(grounding["factors"]):
        coverage.append({**factor,
                         "informative_comparisons": sum(i in pair for pair in informative),
                         "skipped_comparisons": sum(i in pair for pair in skipped)})
    atoms = final["preference_evaluation"]["inferred_atoms"]
    result = {
        "persona_id": record["persona_id"],
        "factor_count": scores.shape[1],
        "unique_score_rows": len(np.unique(scores, axis=0)),
        "identical_option_groups": groups,
        "dominated_item_ids": [ids[i] for i in np.flatnonzero(dominates.any(axis=0))],
        "constant_factor_ids": [grounding["factors"][i]["id"]
                               for i in np.flatnonzero(np.ptp(scores, axis=0) <= 1e-12)],
        "effective_particle_count": float(1 / (weights @ weights)),
        "particle_count": len(weights),
        "maximum_particle_weight": float(weights.max()),
        "prior_confidence": record["rounds"][0]["state_before"]["confidence"],
        "terminal_confidence": state["confidence"],
        "questions_presented": record["questions_presented"],
        "skipped_question_count": record["skipped_question_count"],
        "skip_fraction": record["skipped_question_count"] / record["questions_presented"]
                         if record["questions_presented"] else None,
        "factor_coverage": coverage,
        "preference_summary": final["belief"],
        "preference_evaluation_status": "no_absolute_preference_atoms" if atoms == 0
            else "unresolved" if final["preference_evaluation"]["preference_elicitation_accuracy"] is None
            else "scored",
        "interpretation": "ESS diagnoses particle concentration, not accuracy. Duplicate rows cannot "
            "be distinguished by this utility model. Relative importance alone does not identify "
            "absolute Love/Like/Dislike levels; the shared empty-profile score remains zero.",
    }
    target = record["target"]
    index = ids.index(target)
    result["saved_target_evaluation"] = {
        "target": target, "rank": record["ranking"].index(target) + 1,
        "prior_rank": sorted(range(len(ids)), key=lambda i:
            -record["rounds"][0]["state_before"]["expected_utilities"][i]).index(index) + 1,
        "dominated_by": [ids[j] for j in np.flatnonzero(dominates[:, index])],
        "note": "Evaluation only; the target does not enter inference or parameter selection.",
    }
    return result


def sensitivity(record, config, particle_counts=(500, 2000, 10000),
                kappas=(5., 10., 20., 40.), seed_offsets=(0, 1, 2)):
    """Condition every setting on the same observed pairs; do not simulate new answers."""
    rows = []
    for count in particle_counts:
        for kappa in kappas:
            for offset in seed_offsets:
                seed = record["particle_seed"] + offset
                model = DecisionModel(record["grounding"]["matrix"],
                                      {**config, "particles": count, "kappa": kappa}, seed)
                for snapshot in record["rounds"]:
                    if snapshot["action"] == "ask":
                        model.update(*snapshot["factor_indices"], snapshot["observed_response"]["answer"])
                ranking = model.ranking(record["grounding"]["item_ids"])
                rows.append({"particles": count, "kappa": kappa, "seed": seed,
                             "target_rank": ranking.index(record["target"]) + 1,
                             "ranking": ranking,
                             "effective_particle_count": float(1 / (model.weights @ model.weights)),
                             "confidence": float(model.decision_distribution().max())})
    return {"scope": "Offline fixed-history sensitivity, not an adaptive rerun. Different settings "
                     "would choose different questions; these results do not estimate their online "
                     "performance and must not be used to select defaults on this user's target.",
            "settings": rows}


def verify_replay(record, config):
    """Recompute saved acquisition and Bayesian states without using labels."""
    model = DecisionModel(record["grounding"]["matrix"], config, record["particle_seed"])
    excluded, errors, selections = [], [], []
    for snapshot in record["rounds"]:
        for field, actual in model.state().items():
            errors.append(float(np.max(np.abs(np.asarray(actual) -
                                              np.asarray(snapshot["state_before"][field])))))
        if snapshot["action"] == "stop":
            break
        candidates = model.question_scores(excluded)
        selected = max(candidates, key=lambda candidate: candidate["eig_bits"])
        selections.append(selected["factor_indices"] == snapshot["factor_indices"])
        pair = snapshot["factor_indices"]
        excluded.append(pair)
        if snapshot["action"] == "ask":
            model.update(*pair, snapshot["observed_response"]["answer"])
    return {
        "max_absolute_state_error": max(errors),
        "all_selected_pairs_match": all(selections),
        "ranking_matches": model.ranking(record["grounding"]["item_ids"]) ==
            record["rounds"][-1].get("acquisition_ranking_details", {}).get("ranking", record["ranking"]),
        "initial_particles_match": bool(np.array_equal(model.particles, record["initial_particles"])),
    }


def load_records(run):
    records = []
    runtime_signature = shared.fingerprint(json.loads((run / "runtime.json").read_text()))
    for path in sorted((run / "users").glob("*.json")):
        saved = json.loads(path.read_text())
        if (saved["runtime_signature"] != runtime_signature
                or saved["record_hash"] != shared.fingerprint(saved["record"])):
            raise ValueError(f"Corrupt DECISIVE record: {path}")
        records.append(saved["record"])
    if not records:
        raise ValueError("No completed DECISIVE records")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads((args.run / "manifest.json").read_text())["configuration"]
    if config["method"] != "decisive":
        raise ValueError("Expected a DECISIVE run")
    records = load_records(args.run)
    shared.write_json_atomic(args.output, {
        "source_run": str(args.run.resolve()), "api_calls": 0,
        "users": [{**record_diagnostics(r),
                   "saved_run_replay": verify_replay(r, config["decisive_configuration"]),
                   "parameter_sensitivity": sensitivity(r, config["decisive_configuration"])}
                  for r in records]})
    print(f"Offline DECISIVE diagnostics: {args.output}")


if __name__ == "__main__":
    main()
