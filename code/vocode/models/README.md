# vocode.models: The Architecture Modules

`models/` holds one directory per implemented architecture together with the shared contracts they satisfy. Eleven architecture directories cover the thirteen registered configuration names: the `hifigan/` directory serves the V1, V2, and V3 configurations, and every other directory serves one name.

Two modules define the shared contracts. `vocoder.py` defines `Vocoder`, the module contract every architecture satisfies on top of the framework `Module`: conditioning-to-waveform synthesis, the training recipe, and the measurement interface. The same module defines `PublishedWeights` and `PublishedWeightProvenance`, the author-release contract consisting of the recorded source, the retrieval location, the expected SHA-256 digest, and the strict-load semantics. `registry.py` defines `ModelRegistry`, the closed vocabulary of configuration names and their resolution through `ArchitectureImplementationRecord`, `ArchitectureModuleSpec`, and `ModuleBuildOptions` into architecture modules with their native data defaults.

| Directory | Configurations | Family | Author-weight adapter |
|---|---|---|---|
| `apnet2/` | APNet2 | iSTFT GAN | Yes |
| `bigvgan/` | BigVGAN-base | Waveform GAN | Yes |
| `freev/` | FreeV | iSTFT GAN | Yes |
| `hifigan/` | HiFi-GAN V1, V2, V3 | Waveform GAN | Yes |
| `hiftnet/` | HiFTNet | iSTFT GAN | Yes |
| `lpcnet/` | LPCNet | Autoregressive | No |
| `melgan/` | MelGAN | Waveform GAN | Yes |
| `rfwave/` | RFWave | Rectified flow | No |
| `rndvoc/` | RNDVoC | iSTFT GAN | No |
| `vocos/` | Vocos | iSTFT GAN | Yes |
| `vocosformer/` | VocosFormer | iSTFT GAN with attention | No |

Twelve configurations form the trained cohort of the study. HiFTNet is implemented and adapter-equipped but is a documented non-executed exclusion, not a thirteenth result.

Where a published release exists, the adapter lives beside its architecture as `weights.py` and owns source provenance, retrieval, SHA-256 validation, upstream state extraction, parameter-key adaptation, and strict compatibility loading. Author-released weights are evaluation state and never training initialization, and unsupported author-weight paths fail explicitly rather than falling back to project-trained state.

## Related Components

Each architecture's training objective lives in `../losses/`, and its conditioning front end lives in `../transforms/`. The mirrored tests live in `../../tests/vocode/models/`.
