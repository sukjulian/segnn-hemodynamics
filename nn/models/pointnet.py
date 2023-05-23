import torch
from torch_geometric.nn import MLP, PointNetConv
from torch_scatter import scatter


class FastPointNet(torch.nn.Module):
    def __init__(self, num_channels_in, num_channels_out):
        super().__init__()

        nlc = (32, 64, 128, 212, 256)  # number of latent channels
        kwargs = {'plain_last': False}

        self.sa0_conv = PointNetConv(MLP([num_channels_in + 3, nlc[0], nlc[0], nlc[1]], **kwargs), add_self_loops=False)
        self.sa1_conv = PointNetConv(MLP([nlc[1] + 3, nlc[1], nlc[1], nlc[2]], **kwargs), add_self_loops=False)
        self.sa2_conv = PointNetConv(MLP([nlc[2] + 3, nlc[2], nlc[2], nlc[3]], **kwargs), add_self_loops=False)
        self.sa3_conv = PointNetConv(MLP([nlc[3] + 3, nlc[3], nlc[3], nlc[4]], **kwargs), add_self_loops=False)
        self.sa4_conv = PointNetConv(MLP([nlc[4] + 3, nlc[4], nlc[4], nlc[4]], **kwargs), add_self_loops=False)

        self.fp4_mlp = MLP([nlc[4] + nlc[4], nlc[4], nlc[4]], **kwargs)
        self.fp3_mlp = MLP([nlc[4] + nlc[3], nlc[3], nlc[3]], **kwargs)
        self.fp2_mlp = MLP([nlc[3] + nlc[2], nlc[3], nlc[3]], **kwargs)
        self.fp1_mlp = MLP([nlc[3] + nlc[1], nlc[3], nlc[2]], **kwargs)
        self.fp0_mlp = MLP([nlc[2] + num_channels_in, nlc[2], nlc[2], nlc[2]], **kwargs)

        self.mlp = MLP([nlc[2], num_channels_out])

        print(f"PointNet++ ({sum(param.numel() for param in self.parameters() if param.requires_grad)} parameters)")

    @staticmethod
    def sa(conv, x, pos, data, scale_id):
        sample_idcs = data[f'scale{scale_id}_sample_index']

        edge_index = torch.cat((data[f'scale{scale_id}_pool_source'][None, :], data[f'scale{scale_id}_pool_target'][None, :]), dim=0)

        return conv((x, x[sample_idcs.long()]), (pos, pos[sample_idcs.long()]), edge_index.long()), pos[sample_idcs.long()]

    @staticmethod
    def fp(mlp, x, x_copycat, pos_source, pos_target, data, scale_id):
        pos_diff = pos_source[data[f'scale{scale_id}_interp_source'].long()] - pos_target[data[f'scale{scale_id}_interp_target'].long()]
        squared_pos_distance = torch.clamp(torch.sum(pos_diff ** 2, dim=-1, keepdim=True), min=1e-16)

        x = scatter(
            x[data[f'scale{scale_id}_interp_source'].long()] / squared_pos_distance,
            data[f'scale{scale_id}_interp_target'].long(),
            dim=0,
            reduce='sum'
        ) / scatter(
            1. / squared_pos_distance,
            data[f'scale{scale_id}_interp_target'].long(),
            dim=0,
            reduce='sum'
        )

        return mlp(torch.cat((x, x_copycat), dim=-1))

    def forward(self, data):

        x0, pos0 = self.sa(self.sa0_conv, data.x, data.pos, data, scale_id=0)

        x1, pos1 = self.sa(self.sa1_conv, x0, pos0, data, scale_id=1)
        x2, pos2 = self.sa(self.sa2_conv, x1, pos1, data, scale_id=2)
        x3, pos3 = self.sa(self.sa3_conv, x2, pos2, data, scale_id=3)
        x, pos4 = self.sa(self.sa4_conv, x3, pos3, data, scale_id=4)

        x = self.fp(self.fp4_mlp, x, x3, pos4, pos3, data, scale_id=4)
        x = self.fp(self.fp3_mlp, x, x2, pos3, pos2, data, scale_id=3)
        x = self.fp(self.fp2_mlp, x, x1, pos2, pos1, data, scale_id=2)
        x = self.fp(self.fp1_mlp, x, x0, pos1, pos0, data, scale_id=1)

        x = self.fp(self.fp0_mlp, x, data.x, pos0, data.pos, data, scale_id=0)

        return self.mlp(x)
