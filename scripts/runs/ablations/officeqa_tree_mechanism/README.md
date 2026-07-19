# OfficeQA one-epoch PatchTree mechanism experiment

The launcher runs the full OfficeQA train50 as one batch and one optimization
step. Use the same script with one variant on each resource-pool GPU:

```bash
OFFICEQA_TREE_VARIANT=flat \
  bash scripts/runs/ablations/officeqa_tree_mechanism/run_one_epoch.sh

OFFICEQA_TREE_VARIANT=bottom_up \
  bash scripts/runs/ablations/officeqa_tree_mechanism/run_one_epoch.sh

OFFICEQA_TREE_VARIANT=full \
  bash scripts/runs/ablations/officeqa_tree_mechanism/run_one_epoch.sh
```

The variants are:

- `flat`: PatchRecords directly to Root (`depth=1`);
- `bottom_up`: Cluster to Leaf to Mid to Root, with fallback off;
- `full`: the same depth-3 construction, with direct Root-child evaluation
  and deterministic reconciliation when the Root gate rejects.

All runs use train50, full val24, full test172, three rollout repeats,
`min_support=1`, no Leaf cap in practice, and no tail bank.

Render a command without starting vLLM or training:

```bash
DRY_RUN=1 START_VLLM=0 OFFICEQA_TREE_VARIANT=full \
  bash scripts/runs/ablations/officeqa_tree_mechanism/run_one_epoch.sh
```

These are three end-to-end runs with matched seeds and settings. They do not
yet replay one physically identical PatchRecord JSON across variants; use the
saved `type_guided_v2_patch_records.json` artifacts for the stricter fixed-input
counterfactual analysis after this initial signal test.
