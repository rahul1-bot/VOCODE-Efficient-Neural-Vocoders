# Documentation

This directory carries the report, the final presentation, and the technical documentation of the VOCODE study. The repository root `README.md` is the entry point; these documents provide the depth behind it.

## Report and Presentation

| Path | Contents |
|---|---|
| `report/report.pdf` | The fifteen-page report in JMLR format, the primary written artifact of the study. |
| `report/report.tex` | The main LaTeX source, exactly as built. |
| `report/sections/` | The six section sources with the generated tables and figures they reference. |
| `report/assets/research_papers/references.bib` | The bibliography. |
| `presentation/final_presentation.pdf` | The final project presentation delivered for this course. |

## Guides

| Document | Contents |
|---|---|
| `guides/overview.md` | The study at a glance: the research questions and the principal findings. |
| `guides/architecture.md` | The software design: the framework and client boundary, the atomic command-line interface, capsule anatomy, and the data flow from command to register. |
| `guides/data.md` | LJSpeech, the identifier-sorted partition, the conditioning front ends, and the training-only transformations. |
| `guides/configurations.md` | The twelve-model catalog: families, conditioning, parameters, objectives, optimizers, budgets, and the checkpoint gate. |
| `guides/training.md` | The training procedure: the harness, the recorded seed, durable checkpoints, the registered selection gate, and the training hardware. |
| `guides/metrics.md` | The measurement stack: the quality proxies, the diagnostics, the true-length rule, and the timing and footprint protocol. |
| `guides/deployment.md` | The deployment interventions: the variant vocabulary, per-technique mechanics, per-architecture support, and the admission discipline. |
| `guides/experiments.md` | The executed surface: the three studies, the 91 groups over 273 executions, the decision rule, and the outcome composition. |
| `guides/evidence.md` | The verification guide: register schemas and worked examples tracing report numbers to exact files in `../artifacts/`. |
| `guides/cloud.md` | Cloud execution on Modal: the image, the volume contract, the resource lanes, and the launch workflow. |
| `guides/reproduction.md` | The end-to-end walkthrough from installation to a measured evidence capsule. |

A first reading follows `overview.md`, then `architecture.md`, then the study-definition documents (`data.md`, `configurations.md`, `training.md`), then the measurement documents (`metrics.md`, `deployment.md`, `experiments.md`), and finally `evidence.md` for verification against the shipped registers.
