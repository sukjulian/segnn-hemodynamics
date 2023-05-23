from torch_geometric.nn import fps
from torch_cluster import radius, knn


class PointNetSampling():
    """Computes the hierarchy of vertex subsets and edges used by PointNet. For correct batch creation,
    "torch_geometric.data.Data.__inc__()" has to be overridden.

    Args:
        sample_ratios (tuple): Relative ratios for iterative farthest point sampling.
        edge_radii (tuple): Maximum edge radii determining filter support w.r.t. each scale.
    """

    def __init__(self, sample_ratios, edge_radii):
        self.sample_ratios = sample_ratios
        self.edge_radii = edge_radii

    def __call__(self, data):
        pos = data.pos
        for i, (ratio, r) in enumerate(zip(self.sample_ratios, self.edge_radii)):

            sample_idcs = fps(pos, ratio=ratio)
            sample_idcs, _ = sample_idcs.sort()  # increases stability

            pool_target, pool_source = radius(pos, pos[sample_idcs], r=r)
            interp_target, interp_source = knn(pos[sample_idcs], pos, k=3)

            data[f'scale{i}_pool_target'], data[f'scale{i}_pool_source'] = pool_target.int(), pool_source.int()
            data[f'scale{i}_interp_target'], data[f'scale{i}_interp_source'] = interp_target.int(), interp_source.int()
            data[f'scale{i}_sample_index'] = sample_idcs.int()

            pos = pos[sample_idcs]

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(sample_ratios={self.sample_ratios}, edge_radii={self.edge_radii})"
