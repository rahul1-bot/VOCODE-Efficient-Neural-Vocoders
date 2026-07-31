# Evidence

`artifacts/` is the complete experimental log of the study. Every number in the report traces to a file in this tree, and this guide states where. Three disciplines govern the evidence: every measured row identifies its category and categories are never mixed in one aggregate; experimental records are never edited after the fact; and negative, unsupported, and excluded outcomes remain visible in exclusion registers with the same evidence standard as positive outcomes.

## Layout

```
artifacts/
├── study_1/                     Training reproduction (RQ1)
│   ├── figures/                 Baseline quality, quality-parameters, quality-speed planes
│   └── models/<architecture>/
│       ├── experiments.csv      Per-run summary rows for this model
│       ├── train/               config.yaml, execution.log, per-step metrics.csv
│       ├── test/seed_{42,43,44}/  config.yaml, execution.log, per-seed metrics.csv
│       └── figures/             Training convergence figure
├── study_2/                     Deployment optimization (RQ2)
│   ├── results.csv              One row per admitted group (91 rows)
│   ├── exclusions.csv           One row per exclusion record (29 rows)
│   ├── figures/                 Classification matrix, intervention landscape, technique summary
│   └── models/<architecture>/   experiments.csv, statistics.csv, train/, test/, figures/
└── study_3/                     Cross-model synthesis (RQ3)
    ├── results.csv              One row per configuration (12 rows)
    ├── interventions.csv        The intervention register (91 rows)
    ├── intervention_exclusions.csv  The exclusion register (29 rows)
    ├── frontiers.csv            Frontier membership per objective (9 rows)
    ├── figures/                 Quality-speed, quality-parameters, quality-size, hardware planes
    └── models/<architecture>/   baseline.csv, interventions.csv, figures/
```

## Register Schemas

`study_2/results.csv` carries one row per admitted group: identity (`architecture_name`, `variant_name`, `lane`, `baseline_variant`, `evaluation_seeds`, `seed_count`, `test_utterances`), quality (`pesq`, `stoi`, `mel_error`, `multi_resolution_stft_error`, `mcd`, `utmos_strong`), timing and footprint (`real_time_factor`, `latency_p50_ms`, `latency_p95_ms`, `peak_host_memory_megabytes`, `session_build_seconds`, `deployable_size_megabytes`), the paired contrast (`baseline_pesq`, `pesq_delta`, `rtf_speedup`, `size_ratio`), and the outcome (`classification`, `classification_basis`). `study_3/interventions.csv` shares this schema.

`study_2/exclusions.csv` and `study_3/intervention_exclusions.csv` carry one row per exclusion record: `category`, `architecture_name`, `variant_name`, `attempt_count`, `seeds`, `run_ids`, a machine-readable `reason_code`, and a full-sentence `reason`. The 29 rows document 88 attempts; the `attempt_count` column sums to exactly 88.

`study_3/results.csv` carries one row per configuration: identity and family, `parameters`, `size_megabytes`, recorded and true-length PESQ and STOI, the diagnostics, `utmos_strong`, the timing records (`rtf_b200`, `rtf_m3_max`, `rtf_cpu`, where the Apple-silicon column is a retained comparative record not used by the report), and the frontier flags (`speed_quality_frontier`, `parameter_quality_frontier`, `size_quality_frontier`). `study_3/frontiers.csv` lists the nondominated members per objective with their cost coordinates.

Per-model files add depth: `study_1/models/<architecture>/experiments.csv` holds the per-run summary rows; `study_2/models/<architecture>/experiments.csv` holds the full per-run measurement rows (45 columns including per-metric failure counts, warm-up and repetition counts, and MAC status); and `study_2/models/<architecture>/statistics.csv` holds the paired-contrast statistics per metric (mean delta, standard deviation, standard error, and the confidence interval of the fixed-set analysis).

## Tracing Report Numbers to Files

1. The report states that RNDVoC attained the strongest true-length PESQ, 3.895. In `study_3/results.csv`, the `rndvoc` row carries `pesq_true_length` 3.895272 with `parameters` 3572361 and `size_megabytes` 14.944496; the three per-seed values behind it sit in `study_1/models/rndvoc/test/seed_{42,43,44}/metrics.csv`.
2. The report states that ONNX FP32 accelerated HiFi-GAN V2 by 2.071 times at a PESQ change of negative 0.00013. In `study_2/results.csv`, the row `hifigan_v2, onnx_fp32, cpu` carries `rtf_speedup` 2.0706, `pesq_delta` negative 0.000129, `size_ratio` 0.9822, and `classification` beneficial, with the governing rule sentence in `classification_basis`.
3. The report states that 29 exclusion records document 88 rejected attempts. `study_2/exclusions.csv` holds exactly 29 rows whose `attempt_count` sums to 88; the first row records the APNet2 recovery attempt excluded under reason code `zero_step_recovery_source` because the evaluation used the zero-step unrecovered control rather than the completed 13-epoch recovery state.

The same procedure verifies any reported value: locate the group's row in the study register, then descend into the per-model directory for the per-run and per-seed records behind it.
