You build the root candidate from leaf or mid-level child nodes.

The root is an executable Skill update. Its shared_core and every valid
conditional_residual will be compiled into edits and applied together if the root
candidate passes the normal validation gate.

Rules:
1. The root must perform a genuine abstraction step rather than concatenate,
   summarize, or lightly paraphrase child patches. Extract the invariant reasoning
   operation, decision procedure, or repair principle shared by the covered children.
   The result must remain operational and testable, not vague advice.
2. shared_core contains only a high-level repair mechanism supported across every
   child named in source_child_ids. Remove child-specific entities, narrow triggers,
   answer forms, and local wording from the core.
3. Deduplicate equivalent child instructions and reconcile overlapping target
   regions. Keep distinct mechanisms separate instead of forcing a false common rule.
4. Preserve reusable child-specific behavior as conditional_residuals with explicit
   activation conditions; never make those rules unconditional.
5. Map every child's important constraints in preserved_constraints.
6. Record unresolved conflicts explicitly and do not compile them into the Skill.
7. Do not invent behavior or include sample-specific content.
8. A conditional component may not use delete. target is required except for append.
9. Do not output edits; the program compiles the structure into executable edits.

Respond ONLY with:
{
  "reasoning": "<root abstraction rationale>",
  "shared_core": {
    "condition": "<common activation condition, or empty>",
    "boundary": "<common boundary>",
    "source_child_ids": ["M1", "M2"],
    "patch": {"op": "append|insert_after|replace|delete", "target": "<if needed>", "content": "<shared markdown rule>"}
  },
  "conditional_residuals": [
    {
      "condition": "<child-specific activation condition>",
      "boundary": "<child-specific boundary>",
      "source_child_ids": ["M1"],
      "patch": {"op": "append|insert_after|replace", "target": "<if needed>", "content": "<conditional markdown rule>"}
    }
  ],
  "preserved_constraints": {"M1": ["<retained constraint>"]},
  "unresolved_conflicts": []
}

If no common abstraction is safe, set shared_core to null and retain safe child
rules as conditional_residuals.
