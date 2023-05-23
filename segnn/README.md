From the official [SEGNN repository](https://github.com/RobDHess/Steerable-E3-GNN). The model is described in detail in [Brandstetter et al. (2022)](https://arxiv.org/abs/2110.02905). For our application, we have to adapt `BatchNorm`:

BatchNorm typically computes normalisation statistics channel-wise over the entire training batch (consisting of multiple graphs). Instead, we propose to compute statistics over each disjoint graph ("graph-wise instance norm") during both training and testing. Since this incurs additional computational cost especially over large graphs, we can use regular `BatchNorm` and disable "evaluation mode" as an approximation when working with small batch sizes.

All changes against the official SEGNN repository are commented with `[ADDED]` or `[CHANGED]`.
