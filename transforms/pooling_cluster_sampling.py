import torch
from torch_cluster import radius_graph
from torch_geometric.nn import knn, fps
import numpy as np


class PoolingClusterSampling(object):
    """Compute a hierarchy of vertex clusters using farthest point sampling. For correct batch creation, overwrite
    "torch_geometric.data.Data.__inc__()".

    Args:
        ratios (list): Ratios for farthest point sampling relative to the previous scale hierarchy.
        radii (list): Maximum radii for the creation of radius graphs on each scale.
        loop (bool): Whether to construct self-loop edges.
        max_neighbours (int): Maximum number of neighbours for the radius graphs.
    """

    def __init__(self, ratios, radii, loop=False, max_neighbours=32):
        self.ratios = ratios
        self.radii = radii
        self.loop = loop
        self.max_neighbours = max_neighbours

        self.args = (self.ratios, self.radii, self.loop, self.max_neighbours)

    def __call__(self, data):

        vertices = data.pos
        for i, (ratio, radius) in enumerate(zip(self.ratios, self.radii)):

            if ratio == 1:
                cluster = torch.arange(vertices.shape[0])  # trivial cluster

                if 'edge_index' in data:
                    edges = data.edge_index
                    delattr(data, 'edge_index')
                else:
                    edges = radius_graph(vertices, radius, loop=self.loop, max_num_neighbors=self.max_neighbours)

                indices = torch.arange(vertices.shape[0])  # trivial indices

            else:
                indices = fps(vertices, ratio=ratio)
                indices, _ = indices.sort()  # increases stability

                cluster = knn(vertices[indices], vertices, k=1)[1].numpy()

                unique, cluster = torch.unique(torch.from_numpy(cluster), return_inverse=True)
                vertices = vertices[indices[unique]]

                edges = radius_graph(vertices, radius, loop=self.loop, max_num_neighbors=self.max_neighbours)

                indices = indices[unique]

            data['scale' + str(i) + '_cluster_map'] = cluster.int()
            data['scale' + str(i) + '_edge_index'] = edges.int()
            data['scale' + str(i) + '_sample_index'] = indices.int()

        return data

    def __repr__(self):
        return '{}(ratios={}, radii={}, loop={}, max_neighbours={})'.format(self.__class__.__name__, *self.args)
