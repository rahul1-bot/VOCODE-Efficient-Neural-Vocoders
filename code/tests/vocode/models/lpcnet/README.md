# tests.vocode.models.lpcnet: The LPCNet Tests

This directory holds 83 tests mirroring `vocode/models/lpcnet/`.

The module `lpcnet.py` tests the vocoder module: configuration validation, network construction, the teacher-forced training recipe, and the free-running synthesis interface. The module `network.py` tests the excitation model: the fractional input embedding, the recurrent core with its dual fully connected output, the hierarchical tree probabilities, and the linear-prediction path. The module `sampling.py` tests the sampler: sample-by-sample free-running synthesis, excitation drawing, and waveform reconstruction from excitation and prediction. The module `sparsification.py` tests the scheduled recurrent-weight sparsification: schedule progression between its update bounds, the sparsity targets, the block semantics, and weight clipping.
