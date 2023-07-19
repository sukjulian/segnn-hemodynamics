import torch
import h5py
from tqdm import tqdm
from torch_geometric.nn import fps
from torch_cluster import knn
from prettytable import PrettyTable
from torch_scatter import scatter


def main():

    # model_name, sampling_ratios = "pointnet++", (0.2, 0.25, 0.25, 0.25, 0.25)
    model_name, sampling_ratios = "segnn", (0.04, 0.1)

    print(compute_kernel_radii_statistics("lumen-dataset/raw/lumen_tiny.hdf5", sampling_ratios, model_name, average_num_neighbours=13))
    print("In our experiments, the 92-percentile radii worked well.")


def compute_kernel_radii_statistics(path_to_hdf5, sampling_ratios, model_name, average_num_neighbours):
    cummulative_sampling_ratios = torch.cumprod(torch.tensor(sampling_ratios), 0).tolist()

    with h5py.File(path_to_hdf5, 'r') as hdf5_file:
        sample_ids = list(hdf5_file)

    edge_lengths_dict = {ratio: {'edge_lengths': [], 'scatter_index': []} for ratio in cummulative_sampling_ratios}
    for i, sample_id in enumerate(tqdm(sample_ids)):

        with h5py.File(path_to_hdf5, 'r') as hdf5_file:
            pos = torch.from_numpy(hdf5_file[sample_id]['pos_tets'][()])

        for sampling_ratio, cummulative_sampling_ratio in zip(sampling_ratios, cummulative_sampling_ratios):

            sampling_idcs = fps(pos, ratio=sampling_ratio)
            sampling_idcs, _ = sampling_idcs.sort()  # increases stability

            pos_coarse = pos[sampling_idcs]

            if model_name == "pointnet++":
                target_idcs, source_idcs = knn(pos, pos_coarse, k=average_num_neighbours)

            else:
                target_idcs, source_idcs = knn(pos_coarse, pos_coarse, k=average_num_neighbours)
                pos = pos_coarse

            edge_lengths = torch.norm(pos_coarse[target_idcs] - pos[source_idcs], dim=-1)
            scatter_index = torch.tensor(i).expand(edge_lengths.numel())

            edge_lengths_dict[cummulative_sampling_ratio]['edge_lengths'].append(edge_lengths)
            edge_lengths_dict[cummulative_sampling_ratio]['scatter_index'].append(scatter_index)

            pos = pos_coarse

    for key in edge_lengths_dict.keys():
        edge_lengths_dict[key] = {sub_key: torch.cat(value) for sub_key, value in edge_lengths_dict[key].items()}

    return get_statistics_table(edge_lengths_dict)


def get_statistics_table(edge_lengths_dict):
    percentile = 92

    table = PrettyTable(["Cummulative sampling ratio", "Sample mean", "Mean", f"{percentile}-percentile", "Maximum"])

    table.add_rows([[f"{statistic:.6f}" for statistic in (
        key,
        torch.mean(scatter(value['edge_lengths'], value['scatter_index'], reduce='mean')),
        torch.mean(value['edge_lengths']),
        torch.quantile(value['edge_lengths'], percentile / 100),
        torch.max(value['edge_lengths'])
    )] for key, value in edge_lengths_dict.items()])

    return table


if __name__ == '__main__':
    main()
