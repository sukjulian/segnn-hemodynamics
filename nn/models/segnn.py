from e3nn.o3 import Irreps, spherical_harmonics
import torch
from torch_scatter import scatter

from segnn.o3_building_blocks import O3TensorProduct
from segnn.segnn import SEGNNLayer
from nn import ClusterPooling
from segnn.o3_building_blocks import O3TensorProductSwishGate, O3TensorProduct


class O3Transform():
    def __init__(self, lmax_attr):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, pos, edge_index):

        pos_diff = pos[edge_index[0].long()] - pos[edge_index[1].long()]
        edge_length = torch.linalg.norm(pos_diff, dim=-1, keepdim=True)

        edge_attr = spherical_harmonics(self.attr_irreps, pos_diff, normalize=True, normalization='integral')
        node_attr = scatter(edge_attr, edge_index[1].long(), dim=0, reduce="mean")

        additional_message_features = edge_length

        return edge_attr, node_attr, additional_message_features


class SEGNN(torch.nn.Module):
    def __init__(self, input_irreps, hidden_irreps, output_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps):
        super().__init__()

        self.o3_transform = O3Transform(lmax_attr=1)

        self.embedding_layer = O3TensorProduct(input_irreps, hidden_irreps, node_attr_irreps)

        self.layers = torch.nn.ModuleList()
        for layer_input_irreps, norm in zip((*5 * [hidden_irreps], *2 * [2 * hidden_irreps], hidden_irreps), (*7 * ['batch'], None)):
            self.layers.append(SEGNNLayer(
                layer_input_irreps,
                hidden_irreps,
                hidden_irreps,
                edge_attr_irreps,
                node_attr_irreps,
                norm=norm,
                additional_message_irreps=additional_message_irreps
            ))

        self.pool_0_to_1 = ClusterPooling(target=1)
        self.pool_1_to_2 = ClusterPooling(target=2)

        # [performance: interplation]
        self.unpool_2_to_1 = self.pool_1_to_2.unpool
        self.unpool_1_to_0 = self.pool_0_to_1.unpool

        self.ambient_layers = torch.nn.ModuleList((
            O3TensorProductSwishGate(hidden_irreps, hidden_irreps, node_attr_irreps),
            O3TensorProduct(hidden_irreps, output_irreps, node_attr_irreps)
        ))

        print(f"SEGNN ({sum(param.numel() for param in self.parameters() if param.requires_grad)} parameters)")

    def forward(self, data):
        data.edge_index = data.scale0_edge_index
        scale0_edge_attr, scale0_node_attr, scale0_additional_message_features = self.o3_transform(data.pos, data.edge_index)

        data.x = self.embedding_layer(data.x, scale0_node_attr)

        data.x = self.layers[0](
            data.x,
            data.edge_index.long(),
            scale0_edge_attr,
            scale0_node_attr,
            data.batch,
            scale0_additional_message_features
        )
        copycat0 = data.x.clone()

        # Contracting
        data = self.pool_0_to_1(data)
        scale1_edge_attr, scale1_node_attr, scale1_additional_message_features = self.o3_transform(data.pos, data.edge_index)

        data.x = self.layers[1](
            data.x,
            data.edge_index.long(),
            scale1_edge_attr,
            scale1_node_attr,
            data.batch,
            scale1_additional_message_features
        )
        copycat1 = data.x.clone()

        data = self.pool_1_to_2(data)
        scale2_edge_attr, scale2_node_attr, scale2_additional_message_features = self.o3_transform(data.pos, data.edge_index)

        data.x = self.layers[2](
            data.x,
            data.edge_index.long(),
            scale2_edge_attr,
            scale2_node_attr,
            data.batch,
            scale2_additional_message_features
        )
        data.x = self.layers[3](
            data.x,
            data.edge_index.long(),
            scale2_edge_attr,
            scale2_node_attr,
            data.batch,
            scale2_additional_message_features
        )
        data.x = self.layers[4](
            data.x,
            data.edge_index.long(),
            scale2_edge_attr,
            scale2_node_attr,
            data.batch,
            scale2_additional_message_features
        )

        # Expanding
        data = self.unpool_2_to_1(data)

        data.x = self.layers[5](
            torch.cat((data.x, copycat1), dim=-1),
            data.edge_index.long(),
            scale1_edge_attr,
            scale1_node_attr,
            data.batch,
            scale1_additional_message_features
        )

        data = self.unpool_1_to_0(data)

        data.x = self.layers[6](
            torch.cat((data.x, copycat0), dim=-1),
            data.edge_index.long(),
            scale0_edge_attr,
            scale0_node_attr,
            data.batch,
            scale0_additional_message_features
        )
        data.x = self.layers[7](
            data.x,
            data.edge_index.long(),
            scale0_edge_attr,
            scale0_node_attr,
            data.batch,
            scale0_additional_message_features
        )

        for layer in self.ambient_layers:
            data.x = layer(data.x, scale0_node_attr)

        return data.x
