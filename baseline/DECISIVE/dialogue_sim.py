
import copy

import IQ_recommender as shared
from .. import terminal
from .core import DecisionModel, QUESTION_PROTOCOL, RANKING_PROTOCOL as ACQUISITION_RANKING_PROTOCOL

RANKING_PROTOCOL = shared.RANKING_PROTOCOL
SUMMARY_PROTOCOL = "eig_belief_map_v1"
LEGACY_SUMMARY_PROTOCOL = "decisive_absolute_boundary_summary_v1"

def validate_summary(answer):
    atoms = shared.preference_atoms(answer["description"])
    if any(atom.split(":", 1)[0] not in shared.ABSOLUTE_PREFERENCE_SCORES for atom in atoms):
        raise ValueError("Shared evaluation requires supported absolute preference levels or Unknown")

def tradeoff_question(factors, pair, task="movie"):
    if task not in {"movie", "music"}:
        raise ValueError("task must be movie or music")
    option = "a movie" if task == "movie" else "a music recording to listen to"
    a, b = (factors[i] for i in pair)
    alternatives = {str(i): f"{f['name']}: {f['description']}" for i, f in enumerate((a, b))}
    return {"id": f"decisive_{a['id']}_{b['id']}", "type": "Compare",
            "question": (f"When choosing {option}, which matters more to you: "
                         f"{a['name']} or {b['name']}? Consider the two properties as defined in the options."),
            "options": alternatives}

def run_dialogue(grounding, common, config, seed, calls, answer_fn,
                 saved_rounds=None, on_checkpoint=None, *, catalog):
    task = common.get("task", "movie")
    if [item["item_id"] for item in catalog] != grounding["item_ids"]:
        raise ValueError("Public catalog differs from grounded candidates")
    if grounding.get("task", task) != task:
        raise ValueError("Grounding task differs from dialogue task")
    model = DecisionModel(grounding["matrix"], config, seed)
    saved_rounds = saved_rounds or []
    history, rounds, excluded = [], [], []

    def append(snapshot):
        index = len(rounds)
        if index < len(saved_rounds) and snapshot != saved_rounds[index]:
            raise ValueError("DECISIVE replay differs from the saved round")
        rounds.append(snapshot)
        if on_checkpoint:
            on_checkpoint(rounds)

    while True:
        state = model.state()
        candidates = []
        if state["confidence"] >= config["confidence_threshold"]:
            stop_reason = "decision_confidence"
        elif len(history) >= common["Tmax"]:
            stop_reason = "budget"
        else:
            candidates = model.question_scores(excluded)
            stop_reason = ("pairs_exhausted" if not candidates else
                           "no_information_gain" if max(c["eig_bits"] for c in candidates)
                           <= config["min_eig_bits"] else None)
        snapshot = {"answered_questions": len(history), "history": copy.deepcopy(history),
                    "state_before": state, "candidate_scores": candidates}
        if stop_reason is not None:
            prediction = terminal.terminal_prediction(common, calls, history, catalog)
            belief = prediction["final_preference"]
            snapshot.update(action="stop", stop_reason=stop_reason,
                            **prediction, belief=belief,
                            acquisition_ranking_details={"protocol": ACQUISITION_RANKING_PROTOCOL,
                                             "ranking": model.ranking(grounding["item_ids"]),
                                             "item_ids": grounding["item_ids"],
                                             "expected_utilities": state["expected_utilities"]})
            append(snapshot)
            if len(rounds) < len(saved_rounds):
                raise ValueError("Saved checkpoint extends beyond the current dialogue")
            return {"method": "decisive", "rounds": rounds, "history": history, "T": len(history),
                    "questions_presented": len(excluded),
                    "skipped_question_count": len(excluded) - len(history),
                    "interaction_cost": common["c"] * len(history),
                    "stop_reason": stop_reason, "ranking": snapshot["ranking"],
                    "question_protocol": QUESTION_PROTOCOL, "ranking_protocol": shared.ranking_protocol(common),
                    "evaluation_protocol": shared.EVALUATION_PROTOCOL,
                    "simulator_protocol": shared.SIMULATOR_PROTOCOL,
                    "evaluation_policy": "terminal_only", "particle_seed": seed,
                    **terminal.terminal_metadata(common),
                    "grounding": grounding, "initial_particles": model.particles.tolist()}
        selected = max(range(len(candidates)), key=lambda i: candidates[i]["eig_bits"])
        pair = candidates[selected]["factor_indices"]
        question = tradeoff_question(grounding["factors"], pair, task)
        shared.validate_schema(question, shared.QUESTION)
        response = answer_fn(copy.deepcopy(history), question)
        shared.validate_schema(response, shared.SIMULATOR_RESPONSE)
        skipped = response["answer"] == 2
        if not skipped:
            model.update(*pair, response["answer"])
        snapshot.update(action="skip" if skipped else "ask", selected_index=selected,
                        factor_indices=pair, question=question, observed_response=response,
                        state_after=model.state())
        excluded.append(pair)
        if not skipped:
            history.append({"question": question, "answer": response["answer"]})
        append(snapshot)
