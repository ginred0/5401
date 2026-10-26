"""User-independent, cached factor extraction and ordinal option scoring."""
from ... import llm_recommender as shared
from . import prompts

SCORING_PROTOCOL = "decisive_single_backbone_ordinal_grounding_v1"


class Calls(shared.Calls):
    def ask(self, role, stage, payload, schema, validate=None, **kwargs):
        if stage in {"DECISIVE_FACTORS", "DECISIVE_SCORES"}:
            task = self.cfg.get("task", "movie")
            kwargs["system_prompt"] = prompts.for_task(stage, task)
            if task == "music":
                kwargs["system_prompt"] += "\n" + shared.prompts.MUSIC_DOMAIN
        return super().ask(role, stage, payload, schema, validate, **kwargs)


def build_grounding(catalog, calls, config):
    task = calls.cfg.get("task", "movie")
    decision = prompts.decision_question(task)
    # Explicit allowlist: callers cannot accidentally leak labels or persona data.
    documents = [{k: item[k] for k in ("item_id", "title", "item_description") if k in item}
                 for item in catalog]
    ids = [item["item_id"] for item in documents]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Public candidate IDs must be nonempty and unique")
    factor_schema = shared.array_schema(shared.object_schema(
        name=shared.TEXT, description=shared.TEXT))
    factor_schema.update(minItems=2, maxItems=config["max_factors"])

    def check_factors(answer):
        for field in ("name", "description"):
            keys = [shared.question_key(f[field]) for f in answer["factors"]]
            if not all(keys) or len(set(keys)) != len(keys):
                raise ValueError("Factors must have distinct nonempty names and descriptions")

    answer = calls.ask("recommender", "DECISIVE_FACTORS",
                      {"decision_question": decision, "candidate_items": documents,
                       "max_factors": config["max_factors"]},
                      shared.object_schema(factors=factor_schema), check_factors)
    factors = [{"id": f"F{i+1}", **factor} for i, factor in enumerate(answer["factors"])]
    factor_ids = [f["id"] for f in factors]
    score_schema = shared.object_schema(
        factor_id={"type": "string", "enum": factor_ids},
        level={"type": "string", "enum": [*prompts.LEVELS, "Unknown"]},
        evidence={"type": "string"})
    matrix, assessments = [], []
    for item in documents:
        def check_scores(answer):
            scores = answer["scores"]
            if [s["factor_id"] for s in scores] != factor_ids:
                raise ValueError("Return every factor ID exactly once in factor order")
            sources = [str(item.get(k, "")) for k in ("title", "item_description")]
            for score in scores:
                if score["level"] == "Unknown":
                    if score["evidence"] != "":
                        raise ValueError("Unknown scores must have empty evidence")
                elif not score["evidence"].strip() or not any(
                        score["evidence"] in source for source in sources):
                    raise ValueError("Score evidence must be an exact excerpt from this item's metadata")

        response = calls.ask("recommender", "DECISIVE_SCORES",
                            {"decision_question": decision, "candidate_items": documents,
                             "item_id": item["item_id"], "factors": factors},
                            shared.object_schema(scores=shared.array_schema(score_schema, len(factors))),
                            check_scores)
        assessments.append({"item_id": item["item_id"], **response})
        matrix.append([.5 if score["level"] == "Unknown" else prompts.LEVELS.index(score["level"]) / 7
                       for score in response["scores"]])
    return {"protocol": SCORING_PROTOCOL, "task": task, "item_ids": ids, "factors": factors,
            "matrix": matrix, "assessments": assessments,
            "ordinal_mapping": dict(zip(prompts.LEVELS, [i / 7 for i in range(8)])),
            "unknown_score": .5, "catalog_signature": shared.fingerprint(documents)}
