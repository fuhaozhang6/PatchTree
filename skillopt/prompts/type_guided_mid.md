You build one internal merge-tree node from compatible nodes on the current
frontier. Children may be leaves or lower internal nodes.

The parent must be exactly one abstraction step above its direct children while
retaining every supported conditional detail that would be lost by abstraction.

Rules:
1. Produce one coherent, non-redundant mid-level repair. The shared_core must be
   one abstraction step above the children: express the invariant reasoning
   operation, decision procedure, or repair principle shared by the covered
   leaves.
2. Put only the mechanism genuinely shared by the covered leaves in shared_core.
   Remove child-specific entities, narrow trigger wording, answer forms, and
   sample details from the core. The result must remain operational and testable,
   not vague advice.
3. Deduplicate equivalent child instructions and resolve overlapping target
   regions. Do not mechanically concatenate or paraphrase complete child patches.
4. Put child-specific but reusable behavior in conditional_residuals with explicit
   conditions and source leaf ids.
5. A multi-child parent must be a genuine abstraction: shared_core must be
   non-null, supported by every covered leaf, and more general than the individual
   leaf repairs. Keep the distinguishing details in conditional_residuals.
6. Preserve unique, high-impact leaf insights only when they are not redundant
   with the shared_core. Conditional residuals must not silently broaden their
   source leaf's applicability.
7. If the supplied leaves have no safe common mechanism, do not invent one. Set
   shared_core to null, preserve safe child behavior as residuals, and explain the
   planning incompatibility in unresolved_conflicts.
8. Map every child leaf's important constraints in preserved_constraints.
9. Record incompatible claims in unresolved_conflicts; do not silently average them.
10. Do not introduce behavior unsupported by the children.
11. A conditional component may not use delete. target is required except for append.
12. Do not output edits; the program compiles the node into executable edits.

Respond ONLY with:
{
  "reasoning": "<abstraction rationale>",
  "shared_core": {
    "condition": "<common activation condition, or empty>",
    "boundary": "<common boundary>",
    "source_child_ids": ["L1", "L2"],
    "patch": {"op": "append|insert_after|replace|delete", "target": "<if needed>", "content": "<shared markdown rule>"}
  },
  "conditional_residuals": [
    {
      "condition": "<leaf-specific activation condition>",
      "boundary": "<leaf-specific boundary>",
      "source_child_ids": ["L1"],
      "patch": {"op": "append|insert_after|replace", "target": "<if needed>", "content": "<conditional markdown rule>"}
    }
  ],
  "preserved_constraints": {"L1": ["<retained constraint>"]},
  "unresolved_conflicts": []
}

If no common rule is safe, set shared_core to null and retain the safe child rules
as conditional_residuals.
