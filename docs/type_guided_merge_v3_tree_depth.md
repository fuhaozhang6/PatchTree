# Type-Guided Merge Tree Depth

The runtime supports two bounded shapes:

```text
tree_depth=2: PatchRecord -> Leaf Node -> Root Candidate
tree_depth=3: PatchRecord -> Leaf Node -> Mid Node -> Root Candidate
```

Configuration:

```yaml
optimizer:
  type_guided_merge: true
  type_guided_version: v2
  type_guided_tree_depth: 3
  type_guided_clustering: true
```

## Mid Planning

At depth three, `type_guided_mid_plan` groups compatible leaves. Type labels are
signals rather than hard boundaries. Invalid plans fall back to deterministic
grouping by revision type and repair signature, while unassigned leaves become
singleton mid nodes.

## Node Generation

Leaf, mid, and root nodes all use the same semantic contract:

- `shared_core`: the mechanism supported across covered children;
- `conditional_residuals`: child-specific rules with explicit activation conditions;
- `preserved_constraints`: child-to-parent coverage metadata;
- `unresolved_conflicts`: claims excluded from executable Skill edits;
- `edits`: program-compiled patches produced from the shared core and residuals.

The root receives leaf nodes at depth two and mid nodes at depth three.

## Validation Fallback

There is no leaf self-check. The complete root candidate is evaluated by the
normal validation gate. When rejected, fallback evaluates only the root's direct
children:

```text
depth 2: root rejected -> evaluate leaf children
depth 3: root rejected -> evaluate mid children
```

Accepted child edits are deduplicated/reconciled and validated as one fallback
candidate. This preserves the tree's abstraction level without enumerating all
subsets.

## Artifacts

The merge artifact records `tree_depth`, `root_children_level`,
`root_child_patches`, `leaf_patches`, `mid_plan`, `mid_groups`, `mid_patches`,
and `root_patch`. `type_guided_v2_root.json` contains both semantic node fields
and compiled edits.
