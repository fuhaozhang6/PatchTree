You build one leaf node from compatible PatchRecords.

Extract the mechanism shared by the records into shared_core. Preserve supported
differences as conditional_residuals rather than weakening the shared rule or
flattening a conditional rule into a global instruction.

Rules:
1. shared_core contains only a repair supported by every record it claims to cover.
2. Every conditional_residual must state an explicit activation condition and cite
   its source record ids.
3. Keep each source record's important boundary in preserved_constraints.
4. Put incompatible or unsafe-to-merge claims in unresolved_conflicts.
5. Do not add sample-specific entities, answers, ids, or unsupported behavior.
6. Do not modify protected skill sections.
7. A component patch uses append, insert_after, replace, or delete. A conditional
   component may not use delete. target is required except for append.
8. Do not output edits; the program compiles shared_core and residuals into edits.

Respond ONLY with:
{
  "reasoning": "<merge rationale>",
  "shared_core": {
    "condition": "<common activation condition, or empty when universally applicable>",
    "boundary": "<common non-applicability boundary>",
    "source_child_ids": ["R0001", "R0002"],
    "patch": {"op": "append|insert_after|replace|delete", "target": "<if needed>", "content": "<markdown; omit only for delete>"}
  },
  "conditional_residuals": [
    {
      "condition": "<specific activation condition>",
      "boundary": "<specific boundary>",
      "source_child_ids": ["R0001"],
      "patch": {"op": "append|insert_after|replace", "target": "<if needed>", "content": "<conditional markdown rule>"}
    }
  ],
  "preserved_constraints": {"R0001": ["<constraint retained by the node>"]},
  "unresolved_conflicts": []
}

If there is no rule shared by all records, set shared_core to null and represent
each safe record repair as a conditional_residual.
