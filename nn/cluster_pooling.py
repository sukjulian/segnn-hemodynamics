import torch
from torch_scatter import scatter


class ClusterPooling(torch.nn.Module):

    def __init__(self, target, reduction='mean'):
        super(ClusterPooling, self).__init__()

        self.target = target  # target scale
        self.reduction = reduction

        self.pos_cache = None

        self.batch_cache = None

    def forward(self, data):
        self.pos_cache = data.pos.clone()
        self.batch_cache = data.batch.clone() if 'batch' in data else None

        data.x = scatter(data.x, data['scale' + str(self.target) + '_cluster_map'].long(), dim=0, reduce=self.reduction)
        data.pos = data.pos[data['scale' + str(self.target) + '_sample_index'].long()]
        data.edge_index = data['scale' + str(self.target) + '_edge_index']
        data.batch = data.batch[data['scale' + str(self.target) + '_sample_index'].long()] if 'batch' in data else None

        if hasattr(data, 'h'):
            data.h = scatter(data.h, data['scale' + str(self.target) + '_cluster_map'].long(), dim=0, reduce=self.reduction)
        if hasattr(data, 'edge_align'):
            data.edge_align = data['scale' + str(self.target) + '_edge_align']

        return data

    def unpool(self, data):

        data.x = data.x[data['scale' + str(self.target) + '_cluster_map'].long()]
        data.pos = self.pos_cache
        data.edge_index = data['scale' + str(self.target - 1) + '_edge_index']
        data.batch = self.batch_cache if 'batch' in data else None

        if hasattr(data, 'h'):
            data.h = data.h[data['scale' + str(self.target) + '_cluster_map'].long()]
        if hasattr(data, 'edge_align'):
            data.edge_align = data['scale' + str(self.target - 1) + '_edge_align']

        return data
