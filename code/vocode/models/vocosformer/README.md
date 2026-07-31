# VocosFormer

VocosFormer is the project-defined configuration of the cohort: a literature-derived, capacity-matched Vocos adaptation that inserts a residual-convolution, self-attention, residual-convolution block from the WavTokenizer decoder lineage into a shallower Vocos chassis. It is not a published architecture, and the codec literature is credited for component provenance only. In the cohort it is the 13.78-million-parameter configuration (247,040 parameters or 1.83 percent above Vocos) conditioned on 100 unnormalized HTK-scale mel bands at 24 kHz, trained for 124.8 thousand updates with a retained state at 120 thousand under matched losses, data, and training quantum.

| Module | Contents |
|---|---|
| `vocosformer.py` | `VocosformerConfig` and `Vocosformer`, the vocoder module. |
| `network.py` | `VocosformerNetwork`, the generator. |

`Vocosformer` subclasses the reproduced Vocos module, so the discriminators, the losses, the optimizer and schedule declaration, and the mel protocol are inherited unchanged; the configuration differs only in the generator network it constructs. `VocosformerNetwork` places the inserted block (a residual convolution, a frame-attention block with a position network, and a second residual convolution) inside the Vocos ConvNeXt chassis and reconstructs the waveform through the inherited inverse-STFT head, configured through `VocosformerNetworkConfig`. There is no discriminator module and no author-weight adapter, because both are inherited or inapplicable by construction.

The study measured this matched state against its Vocos control at configuration level. It did not improve on the control, and the gap between its padded and true-length PESQ values is consistent with contamination through unmasked global attention; the report records this as a bounded mechanism observation from one training trajectory.

## Related Components

The inherited training objective lives in `../../losses/vocos.py`, and the parent module lives in `../vocos/`. The mirrored tests live in `../../../tests/vocode/models/vocosformer/`.
