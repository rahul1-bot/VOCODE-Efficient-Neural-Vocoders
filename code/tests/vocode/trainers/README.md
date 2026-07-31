# tests.vocode.trainers: The Execution Role Tests

This directory holds 123 tests mirroring `vocode/trainers/`, with one test module per execution role.

The module `reproduction.py` tests `ReproductionTrainingRunner`: module and DataModule construction from resolved configurations, stage dispatch through the framework trainer, checkpoint evidence flow, and resumability semantics. The module `published.py` tests `PublishedWeightsEvaluator`: the module specification of each adapter-equipped architecture, retrieval and verification through fabricated releases, strict loading, and measurement under the shared protocol; author weights are asserted to be evaluation state only. The module `optimized.py` tests `OptimizedVariantEvaluator`: the coupling of a retained checkpoint with one deployment transformation, lane execution, paired-baseline evidence placement, and the required provenance of the optimized-variants lane.

The roles are tested to write into their own evidence categories, so lane separation is enforced by test rather than by convention alone.
