# Third-party notices

## SpikingBrain

This repository is an independent community project for experimenting with
CPU-only distributed inference for the SpikingBrain model architecture. It is
not the official SpikingBrain project and is not affiliated with, sponsored
by, or endorsed by BICLab or the upstream model authors.

The model architecture and upstream work are attributed to **BICLab /
SpikingBrain-7B**. The checkpoint used in the documented experiments was
**Abel2076/SpikingBrain-7B-W8ASpike**.

This repository does not contain or redistribute model weights, checkpoint
shards, or a copied tokenizer. SpikingBrain, its source materials, and related
checkpoints retain their own copyright notices, licenses, terms, and
attributions. Users must obtain the model and tokenizer separately from their
upstream sources and comply with the terms that apply there.

The MIT license in [LICENSE](LICENSE) applies only to the original software and
documentation in this repository. It does not relicense SpikingBrain, the
checkpoint, PyTorch, OpenBLAS, Transformers, Safetensors, or other third-party
components.

## Runtime dependencies

The project uses third-party software including PyTorch, OpenBLAS, Gloo,
Transformers, Safetensors, pytest, and Docker. Their respective licenses and
notices continue to apply. The stable runtime Dockerfile fetches PyTorch source
from upstream and builds it under PyTorch's own license; the resulting wheel
and Docker image are not committed here.
