# vocode.trainers: The Execution Roles

`trainers/` binds validated configurations to the `syntheticmind` trainer for the three execution roles of the study. The roles consume the same data, metric, and logging stack; they differ only in which weights they execute and which evidence category they write.

| Module | Role |
|---|---|
| `reproduction.py` | `ReproductionTrainingRunner` executes project-trained reproduction. |
| `published.py` | `PublishedWeightsEvaluator` executes author-released checkpoint evaluation. |
| `optimized.py` | `OptimizedVariantEvaluator` executes optimized-variant evaluation. |

`ReproductionTrainingRunner` constructs the architecture module and the DataModule from the resolved configuration, drives training, validation, or evaluation through the framework trainer, and persists checkpoint and result evidence into the reproduction lane. Training runs are resumable from durable checkpoints, and the retained state is selected by the registered checkpoint gate rather than by a retrospective search.

`PublishedWeightsEvaluator` evaluates author releases. A `PublishedWeightsModuleSpec` names the architecture and its weight adapter; the evaluator retrieves and verifies the release through `models/<architecture>/weights.py`, loads it strictly, and measures it under the same protocol as project-trained states. Author weights are evaluation state and are never used as training initialization.

`OptimizedVariantEvaluator` evaluates deployment artifacts. An `OptimizedVariantModuleSpec` couples one Retained Project Checkpoint with one transformation from `../optimization/`; the evaluator applies the transformation, executes the transformed artifact on the requested hardware lane, and writes the measured row into the optimized-variants lane beside its paired baseline.

The three roles map onto the evidence categories of the command-line interface: `project_trained_reproduction`, `published_checkpoint_evaluation`, and `project_optimized_variants`, with `project_hybrid_variants` sharing the reproduction machinery. Measured rows always carry their category, so the lanes are never mixed in one aggregate.

## Related Components

The subcommand-to-role binding is defined in `vocode/cli.py`. The mirrored tests live in `../../tests/vocode/trainers/`.
