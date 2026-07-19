from skillopt.config import flatten_config
from skillopt.engine import trainer
from skillopt.gradient import type_guided_merge_v2


def _node_json(*, content: str, condition: str, source_ids: list[str]) -> str:
    ids = ",".join(f'"{item}"' for item in source_ids)
    return (
        '{"reasoning":"merge","shared_core":{'
        f'"condition":"{condition}","boundary":"",'
        f'"source_child_ids":[{ids}],'
        f'"patch":{{"op":"append","content":"{content}"}}}},'
        '"conditional_residuals":[],"preserved_constraints":{},'
        '"unresolved_conflicts":[]}'
    )


def test_type_guided_v2_config_flattens_from_optimizer_section():
    flat = flatten_config({
        "optimizer": {
            "type_guided_rollout_repeats": 3,
            "type_guided_max_patch_records": 11,
            "type_guided_tree_depth": 3,
            "type_guided_fallback_tau_child": 0.1,
            "type_guided_fallback_reconcile": "llm_select",
            "type_guided_patch_record_workers": 2,
        }
    })
    assert flat["type_guided_rollout_repeats"] == 3
    assert flat["type_guided_max_patch_records"] == 11
    assert flat["type_guided_tree_depth"] == 3
    assert flat["type_guided_fallback_tau_child"] == 0.1
    assert flat["type_guided_fallback_reconcile"] == "llm_select"
    assert flat["type_guided_patch_record_workers"] == 2


def test_type_guided_v2_generates_compact_dataset_specific_record(monkeypatch, tmp_path):
    seen = {}

    def fake_chat(*, system, user, **kwargs):
        seen["system"] = system
        return (
            '{"no_patch":false,"question_type":"short_factoid_answering",'
            '"revision_type":"answer_span_minimization",'
            '"repair_signature":"return minimal answer span",'
            '"condition":"the question asks for a short entity",'
            '"boundary":"retain required qualifiers",'
            '"patch":{"op":"append","content":"Return only the minimal answer span."}}',
            {},
        )

    monkeypatch.setattr(type_guided_merge_v2, "chat_optimizer", fake_chat)
    records, artifact = type_guided_merge_v2.generate_patch_records(
        skill_content="",
        repeated_rollouts=[
            {"repeat_id": 0, "results": [{"id": "q1", "hard": 1, "soft": 1.0}]},
            {"repeat_id": 1, "results": [{"id": "q1", "hard": 0, "soft": 0.0}]},
        ],
        tau_succ=1.0,
        workers=1,
        step_cache_dir=str(tmp_path),
        include_trajectories=False,
        env_name="searchqa",
        verbose=False,
    )
    assert "context-grounded short-answer QA" in seen["system"]
    assert artifact["n_candidates"] == 1
    assert artifact["n_records"] == 1
    assert records == [{
        "question_type": "short_factoid_answering",
        "revision_type": "answer_span_minimization",
        "repair_signature": "return minimal answer span",
        "condition": "the question asks for a short entity",
        "boundary": "retain required qualifiers",
        "patch": {"op": "append", "content": "Return only the minimal answer span."},
        "record_id": "R0001",
    }]


def test_type_guided_v2_skips_stable_success(monkeypatch):
    def fail_chat(**kwargs):
        raise AssertionError("stable successes should not call optimizer")

    monkeypatch.setattr(type_guided_merge_v2, "chat_optimizer", fail_chat)
    records, artifact = type_guided_merge_v2.generate_patch_records(
        skill_content="",
        repeated_rollouts=[
            {"repeat_id": 0, "results": [{"id": "q1", "hard": 1, "soft": 1.0}]},
            {"repeat_id": 1, "results": [{"id": "q1", "hard": 1, "soft": 1.0}]},
        ],
        tau_succ=1.0,
        workers=1,
        include_trajectories=False,
        verbose=False,
    )
    assert records == []
    assert artifact["n_stable_success"] == 1


def test_type_guided_v2_compiles_shared_core_and_conditional_residual(monkeypatch):
    def fake_chat(*, stage, **kwargs):
        if stage == "type_guided_leaf":
            return _node_json(
                content="Compare claims precisely.",
                condition="answer choices make related claims",
                source_ids=["R0001"],
            ), {}
        return (
            '{"reasoning":"root","shared_core":{'
            '"condition":"answer choices differ in logical strength",'
            '"boundary":"not open-ended generation","source_child_ids":["L1"],'
            '"patch":{"op":"append","content":"Match claim strength to the evidence."}},'
            '"conditional_residuals":[{'
            '"condition":"an option adds a universal qualifier","boundary":"",'
            '"source_child_ids":["L1"],'
            '"patch":{"op":"append","content":"Verify the added qualifier explicitly."}}],'
            '"preserved_constraints":{"L1":["do not overgeneralize"]},'
            '"unresolved_conflicts":[]}',
            {},
        )

    monkeypatch.setattr("skillopt.gradient.type_guided_merge.chat_optimizer", fake_chat)
    root, artifact = type_guided_merge_v2.merge_type_guided_v2_records(
        skill_content="",
        patch_records=[{
            "record_id": "R0001",
            "question_type": "strength_comparison",
            "revision_type": "strength_calibration",
            "repair_signature": "match conclusion strength",
            "condition": "answer choices make related claims",
            "boundary": "not open generation",
            "patch": {"op": "append", "content": "Compare claims."},
        }],
        min_support=1,
        verbose=False,
    )
    assert artifact["version"] == "v2"
    assert "self_check" not in artifact
    assert root["shared_core"]["source_child_ids"] == ["L1"]
    assert len(root["conditional_residuals"]) == 1
    assert root["edits"][0]["node_component"] == "shared_core"
    assert root["edits"][0]["content"].startswith(
        "When answer choices differ in logical strength:"
    )
    assert root["edits"][1]["node_component"] == "conditional_residual"
    assert "Verify the added qualifier explicitly." in root["edits"][1]["content"]


def test_type_guided_v2_depth1_passes_records_directly_to_root(monkeypatch):
    stages: list[str] = []

    def fake_chat(*, stage, **kwargs):
        stages.append(stage)
        assert stage == "type_guided_root"
        return _node_json(
            content="Verify the source value before calculating.",
            condition="a numeric answer depends on a table value",
            source_ids=["R0001"],
        ), {}

    monkeypatch.setattr("skillopt.gradient.type_guided_merge.chat_optimizer", fake_chat)
    root, artifact = type_guided_merge_v2.merge_type_guided_v2_records(
        skill_content="",
        patch_records=[{
            "record_id": "R0001",
            "question_type": "derived_numeric_answer",
            "revision_type": "operand_verification",
            "repair_signature": "verify source operands",
            "condition": "a numeric answer depends on a table value",
            "boundary": "",
            "patch": {"op": "append", "content": "Verify the source value."},
        }],
        min_support=1,
        tree_depth=1,
        clustering_enabled=False,
        verbose=False,
    )

    assert stages == ["type_guided_root"]
    assert artifact["tree_depth"] == 1
    assert artifact["root_children_level"] == "record"
    assert artifact["leaf_patches"] == []
    assert artifact["mid_patches"] == []
    assert artifact["root_child_patches"][0]["record_id"] == "R0001"
    assert root["edits"]


def test_type_guided_v2_cluster_type_propagates_to_leaf(monkeypatch):
    def fake_chat(*, stage, **kwargs):
        if stage == "type_guided_cluster":
            return (
                '{"reasoning":"cluster","clusters":[{'
                '"cluster_id":"C1","cluster_label":"entity form",'
                '"question_type":"factoid_named_entity_answering",'
                '"revision_type":"answer_form_enforcement",'
                '"repair_signature":"canonical entity form",'
                '"record_ids":["R0001","R0002"],'
                '"merge_rationale":"same answer form repair","boundary":"not long answers"}],'
                '"singletons":[]}',
                {},
            )
        ids = ["R0001", "R0002"] if stage == "type_guided_leaf" else ["L1"]
        return _node_json(content="Use a canonical entity form.", condition="a short entity is requested", source_ids=ids), {}

    monkeypatch.setattr("skillopt.gradient.type_guided_merge.chat_optimizer", fake_chat)
    monkeypatch.setattr(type_guided_merge_v2, "chat_optimizer", fake_chat)
    root, artifact = type_guided_merge_v2.merge_type_guided_v2_records(
        skill_content="",
        patch_records=[
            {
                "record_id": "R0001",
                "question_type": "ambiguous_entity_reference",
                "revision_type": "entity_disambiguation",
                "repair_signature": "canonical entity form",
                "condition": "a short entity is requested",
                "boundary": "not long answers",
                "patch": {"op": "append", "content": "Use canonical entity names."},
            },
            {
                "record_id": "R0002",
                "question_type": "short_factoid_answering",
                "revision_type": "answer_span_minimization",
                "repair_signature": "canonical entity form",
                "condition": "a short entity is requested",
                "boundary": "not long answers",
                "patch": {"op": "append", "content": "Prefer entity-only answers."},
            },
        ],
        min_support=1,
        clustering_enabled=True,
        cluster_target_size=2,
        cluster_max_size=4,
        verbose=False,
    )
    leaf = artifact["leaf_patches"][0]
    assert root["edits"]
    assert leaf["cluster_question_type"] == "factoid_named_entity_answering"
    assert leaf["cluster_revision_type"] == "answer_form_enforcement"
    assert leaf["member_question_type_counts"] == {
        "ambiguous_entity_reference": 1,
        "short_factoid_answering": 1,
    }


def test_fallback_reconcile_deduplicates_exact_edits():
    child_patches = [
        {"mid_id": "M1", "edits": [{"op": "append", "content": "Keep answers short."}]},
        {"mid_id": "M2", "edits": [{"op": "append", "content": "Keep answers short."}]},
    ]
    edits, report = trainer._reconcile_fallback_edits(
        child_patches=child_patches,
        child_rows=[
            {"child_id": "M1", "gate_score": 0.8},
            {"child_id": "M2", "gate_score": 0.7},
        ],
        update_mode="patch",
        mode="deterministic",
        min_children=2,
    )
    assert edits == [{"op": "append", "content": "Keep answers short."}]
    assert report["n_input_edits"] == 2
    assert report["n_output_edits"] == 1
    assert report["dropped_edits"][0]["reason"] == "exact_duplicate_of:M1_E1"


def test_fallback_reconcile_llm_select_keeps_only_original_edits(monkeypatch):
    def fake_chat(*, stage, **kwargs):
        assert stage == "type_guided_fallback_reconcile"
        return (
            '{"reasoning":"drop broader duplicate","keep_edit_ids":["M2_E1"],'
            '"drop_edit_ids":[{"edit_id":"M1_E1","reason":"weaker_than:M2_E1"}],'
            '"conflict_groups":[]}',
            {},
        )

    monkeypatch.setattr(trainer, "chat_optimizer", fake_chat)
    child_patches = [
        {"mid_id": "M1", "edits": [{"op": "append", "content": "Answer concisely."}]},
        {"mid_id": "M2", "edits": [{"op": "append", "content": "Answer with only the concise entity."}]},
    ]
    edits, report = trainer._reconcile_fallback_edits(
        child_patches=child_patches,
        child_rows=[
            {"child_id": "M1", "gate_score": 0.6},
            {"child_id": "M2", "gate_score": 0.8},
        ],
        update_mode="patch",
        mode="llm_select",
        min_children=2,
    )
    assert edits == [{"op": "append", "content": "Answer with only the concise entity."}]
    assert report["llm_used"] is True
    assert report["status"] == "llm_selected"


def test_validated_frontier_fuse_rewrites_with_preserved_provenance(monkeypatch):
    def fake_chat(*, stage, system, **kwargs):
        assert stage == "type_guided_validated_frontier_fuse"
        assert "must not broaden" in system
        return (
            '{"reasoning":"combine compatible validated rules","edits":[{'
            '"op":"append","content":"Prefer the shortest directly supported entity.",'
            '"condition":"the question requests a short named entity",'
            '"boundary":"a qualifier is required to disambiguate",'
            '"source_child_ids":["M1","M2"]}],'
            '"dropped_child_insights":[]}',
            {},
        )

    monkeypatch.setattr(trainer, "chat_optimizer", fake_chat)
    edits, report = trainer._fuse_validated_frontier_edits(
        skill_content="## Existing Skill",
        child_patches=[
            {
                "mid_id": "M1",
                "support_count": 2,
                "edits": [{"op": "append", "content": "Return a short entity."}],
            },
            {
                "mid_id": "M2",
                "support_count": 3,
                "edits": [{"op": "append", "content": "Keep required qualifiers."}],
            },
        ],
        child_rows=[
            {"child_id": "M1", "gate_score": 0.8, "improvement": 0.1},
            {"child_id": "M2", "gate_score": 0.75, "improvement": 0.05},
        ],
        update_mode="patch",
    )
    assert report["status"] == "llm_fused"
    assert report["llm_used"] is True
    assert report["n_input_edits"] == 2
    assert report["n_output_edits"] == 1
    assert edits[0]["source_child_ids"] == ["M1", "M2"]
    assert edits[0]["support_count"] == 5
    assert edits[0]["content"].startswith(
        "When the question requests a short named entity:"
    )
    assert "Do not apply this rule when a qualifier is required" in edits[0]["content"]


def test_validated_frontier_fuse_falls_back_to_dedup(monkeypatch):
    def fail_chat(**kwargs):
        raise RuntimeError("optimizer unavailable")

    monkeypatch.setattr(trainer, "chat_optimizer", fail_chat)
    duplicate = {"op": "append", "content": "Keep answers short."}
    edits, report = trainer._fuse_validated_frontier_edits(
        skill_content="",
        child_patches=[
            {"mid_id": "M1", "edits": [duplicate]},
            {"mid_id": "M2", "edits": [duplicate]},
        ],
        child_rows=[
            {"child_id": "M1", "gate_score": 0.8},
            {"child_id": "M2", "gate_score": 0.7},
        ],
        update_mode="patch",
    )
    assert edits == [duplicate]
    assert report["status"] == "llm_fuse_failed_dedup_only"
    assert report["llm_used"] is False
    assert report["n_output_edits"] == 1
