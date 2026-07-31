# Package initializer for the experiment runners; modules are imported by
# full path.
#
# This package holds the three roles that drive the harness, one per evidence
# family of the study:
# 1. reproduction.ReproductionTrainingRunner trains the Study 1 cohort from
#    random initialization and evaluates it, and is the only role here that
#    builds a training Trainer or writes a checkpoint
# 2. published.PublishedWeightsEvaluator measures an author-released checkpoint
#    inside the project's own implementation, restricted to the test stage
# 3. optimized.OptimizedVariantEvaluator measures one Study 2 capsule: a
#    project-trained checkpoint transformed by a registered technique
#
# Design decisions:
# - Each role owns what enters the harness while the harness owns the loops, so
#   a policy question (which callbacks, which cadence, which precision) is
#   answered here and a mechanism question is answered in syntheticmind
# - One instance of any role handles exactly one cell and shares no state with
#   another instance, so several cells may run in one process without their
#   evidence interfering
# - The package exports nothing; every role is imported by its full path, which
#   keeps the command-line surface explicit about which evidence family it is
#   about to produce
#
# Author: Rahul Sawhney

__all__: list[str] = []
