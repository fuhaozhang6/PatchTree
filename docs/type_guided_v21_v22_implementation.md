# Type-Guided Tail-Bank Notes

This file previously described the removed raw/anchor/canonical type pipeline.
The current runtime uses dataset-owned PatchRecord prompts and persists only
`question_type`, `revision_type`, `repair_signature`, `condition`, `boundary`,
and `patch` plus the program-assigned `record_id`.

The tail-bank remains available for low-support records:

- collect groups dropped because support is below the step threshold;
- group them across a rolling epoch window by question type, revision type, and
  repair signature;
- optionally require evidence from more than one training step;
- merge selected records through the same leaf/mid/root path;
- apply the tail candidate only when the normal validation gate accepts it.

There is no type canonicalizer, support self-check, or leaf self-check in the
current runtime.
