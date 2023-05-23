from argparse import ArgumentParser

from transforms import PointNetSampling, PoolingClusterSampling
from torch_geometric.transforms import Compose
from datasets import LumenDataset
import wandb_impostor as wandb
import torch_geometric as pyg
from nn.models import FastPointNet as PointNet
from e3nn.o3 import Irreps
from segnn.balanced_irreps import BalancedIrreps
from nn.models import SEGNN
import torch
import os
from scipy.spatial.transform import Rotation
import numpy as np
from torch_cluster import knn
from tqdm import tqdm
import statistics
from utils import AccuracyAnalysis, VTUWriter
import sys
from time import asctime
from torch.nn.parallel import DistributedDataParallel


parser = ArgumentParser()

parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=0)
args = parser.parse_args()

wandb_config = {
    'batch_size': 2,
    'learning_rate': 3e-4,  # best learning rate for Adam, hands down
    'num_epochs': args.num_epochs,
    'loss_term_factors': {'generic': 1.}
}


def main(rank, num_gpus):
    ddp_setup(rank, num_gpus)

    lumen_dataset = LumenDataset("lumen-dataset", pre_transform=Compose([
        # PointNetSampling(sample_ratios=(0.2, 0.25, 0.25, 0.25, 0.25), edge_radii=(0.0183, 0.0321, 0.0530, 0.0892, 0.1559)),
        tets_to_edges,
        PoolingClusterSampling(ratios=(1., 0.04, 0.1), radii=(None, 0.0568, 0.1372)),
        input_transform
    ]), transform=random_rotation)

    training_data_loader = pyg.loader.DataLoader(
        lumen_dataset[get_dataset_slices_for_gpus(num_gpus, num_samples=16)[rank]],
        batch_size=wandb.config['batch_size'],
        shuffle=True,
        # num_workers=16  # may produce error (observed on Slurm cluster)
    )
    validation_data_loader = pyg.loader.DataLoader(
        lumen_dataset[get_dataset_slices_for_gpus(num_gpus, num_samples=2, first_sample_idx=16)[rank]],
        batch_size=wandb.config['batch_size'],
        shuffle=True,
        # num_workers=16
    )
    test_dataset_slice = slice(18, 20)
    visualisation_dataset_range = range(18, 20)

    # neural_network = PointNet(num_channels_in=9, num_channels_out=3)
    neural_network = SEGNN(
        input_irreps=Irreps('3x1o'),
        hidden_irreps=BalancedIrreps(lmax=1, vec_dim=24),
        output_irreps=Irreps('1x1o'),
        edge_attr_irreps=Irreps.spherical_harmonics(lmax=1),
        node_attr_irreps=Irreps.spherical_harmonics(lmax=1),
        additional_message_irreps=Irreps('1x0e')
    )

    training_device = torch.device(f'cuda:{rank}')
    neural_network.to(training_device)

    load_neural_network_weights(neural_network)

    # Distributed data parallel (multi-GPU training)
    neural_network = ddp_module(neural_network, rank)

    loss_function = torch.nn.L1Loss()
    optimiser = torch.optim.Adam(neural_network.parameters(), lr=wandb.config['learning_rate'])

    wandb.watch(neural_network)
    optimisation_loop({
        'neural_network': neural_network,
        'training_device': training_device,
        'optimiser': optimiser,
        'loss_function': loss_function,
        'training_data_loader': training_data_loader,
        'validation_data_loader': validation_data_loader
    })

    ddp_rank_zero(assessment_loop, {
        'neural_network': neural_network,
        'training_device': training_device,
        'dataset': lumen_dataset,
        'test_dataset_slice': test_dataset_slice,
        'visualisation_dataset_range': visualisation_dataset_range
    })

    ddp_cleanup()


def get_dataset_slices_for_gpus(num_gpus, num_samples, first_sample_idx=0):
    per_gpu = int(num_samples / num_gpus)

    first_and_last_idx_per_gpu = tuple(zip(
        range(first_sample_idx, first_sample_idx + num_samples - per_gpu + 1, per_gpu),
        range(first_sample_idx + per_gpu, first_sample_idx + num_samples + 1, per_gpu)
    ))

    return [slice(*idcs) for idcs in first_and_last_idx_per_gpu]


def load_neural_network_weights(neural_network):
    if os.path.exists("neural_network_weights.pt"):

        neural_network.load_state_dict(torch.load("neural_network_weights.pt"))
        print("Resuming from pre-trained neural-network weights.")


def random_rotation(data):
    R = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform(0., 2. * np.pi, size=3)).as_matrix().astype('f4'))

    data.y = data.y @ R.t()
    data.pos = data.pos @ R.t()

    data.x = torch.cat((data.x[:, 0:3] @ R.t(), data.x[:, 3:6] @ R.t(), data.x[:, 6:9] @ R.t()), dim=-1)

    # add translation for SE(3) transform

    return data


def tets_to_edges(data):

    edge_index = torch.cat([data.tets[0:2], data.tets[1:3], data.tets[2:4], data.tets[0:4:3]], dim=1)
    data.edge_index = pyg.utils.to_undirected(edge_index)

    delattr(data, 'tets')

    return data


def input_transform(data):
    with torch.no_grad():

        nearest_boundary_vertex_index = compute_nearest_boundary_vertex(data)
        data.x = torch.cat((
            data.pos[nearest_boundary_vertex_index['inlet'].long()] - data.pos,
            data.pos[nearest_boundary_vertex_index['lumen_wall'].long()] - data.pos,
            data.pos[nearest_boundary_vertex_index['outlets'].long()] - data.pos
        ), dim=-1)

    return data


def compute_nearest_boundary_vertex(data):
    index_dict = {}

    for key in ('inlet', 'lumen_wall', 'outlets'):
        index_dict[key] = data[f'{key}_index'][knn(data.pos[data[f'{key}_index'].long()], data.pos, k=1)[1].long()]

    return index_dict


def optimisation_loop(config):
    for epoch in tqdm(range(wandb.config['num_epochs']), desc="Epochs", position=0, leave=True):

        loss_data = {key: {'generic': []} for key in ('training', 'validation')}

        # Objective convergence
        config['neural_network'].train()

        for batch in tqdm(config['training_data_loader'], desc="Training split", position=1, leave=False):
            config['optimiser'].zero_grad()

            batch = batch.to(config['training_device'])
            prediction = config['neural_network'](batch)

            loss_term_factors =  wandb.config['loss_term_factors']
            loss_terms = {'generic': loss_term_factors['generic'] * config['loss_function'](prediction, batch.y)}
            loss_value = sum(loss_terms.values())

            for key, value in loss_terms.items():
                loss_data['training'][key].append(value.item())

            loss_value.backward()  # "autograd" hook fires and triggers gradient synchronisation across processes
            config['optimiser'].step()

            del batch, prediction

        ddp_rank_zero(torch.save, config['neural_network'].state_dict(), "neural_network_weights.pt")

        # Learning task
        # config['neural_network'].eval()  # training-mode "BatchNorm" approximates "instance norm"

        with torch.no_grad():
            for batch in tqdm(config['validation_data_loader'], desc="Validation split", position=1, leave=False):

                batch = batch.to(config['training_device'])
                prediction = config['neural_network'](batch)

                loss_terms = {'generic': config['loss_function'](prediction, batch.y)}
                loss_value = sum(loss_terms.values())

                for key, value in loss_terms.items():
                    loss_data['validation'][key].append(value.item())

                del batch, prediction

        for phase_name in loss_data.keys():
            wandb.log({phase_name: {key: statistics.median(value) for key, value in loss_data[phase_name].items()}})


def assessment_loop(config):
    accuracy_analysis = {'velocity': AccuracyAnalysis()}
    vtu_writer = VTUWriter()

    if isinstance(config['neural_network'], PointNet):
        config['neural_network'].eval()  # training-mode "BatchNorm" equals "instance norm" (batch size one)

    with torch.no_grad():

        # Quantitative
        for i, data in enumerate(tqdm(config['dataset'][config['test_dataset_slice']], desc="Test split", position=0, leave=False)):

            data = data.to(config['training_device'])
            prediction = config['neural_network'](data)

            accuracy_analysis['velocity'].append_values({
                'ground_truth': data.y.cpu(),
                'prediction': prediction.cpu(),
                'scatter_idx': torch.tensor(i)
            })

            del data

        print(f"Velocity\n{accuracy_analysis['velocity'].accuracy_table()}")

        # Qualitative (visual)
        # config['neural_network'].cpu()  # avoid memory issues

        for idx in tqdm(config['visualisation_dataset_range'], desc="Visualisation split", position=0, leave=False):
            data = config['dataset'].get_geometry_for_visualisation(idx)

            for key, value in config['dataset'].__getitem__(idx):
                data[key] = value

            data = data.to(config['training_device'])  # avoid "Floating point exception"
            prediction = config['neural_network'](input_transform(data.clone()))

            data['U'] = prediction

            data['pool'] = compute_pooling_visuals(data) if 'scale0_sample_index' in data else None
            data['x_inlet'], data['x_lumen_wall'], data['x_outlets'] = data.x[:, 0:3], data.x[:, 3:6], data.x[:, 6:9]
            [data.pop(key) for key in data.keys if 'scale' in key or key == 'x' or key == 'node_attr']
            vtu_writer("visuals_idx_{:06d}.vtu".format(idx), data.to('cpu'))


def compute_pooling_visuals(data):

    pooling_scale_nums = []
    for key in data.keys:

        if 'scale' in key and 'cluster_map' in key:
            pooling_scale_nums.extend([int(string) for string in key if string.isdigit()])

    pooling_scale_nums.sort()

    pooling_visuals_tensor = torch.zeros(data.num_nodes)
    for pooling_scale_num in pooling_scale_nums:
        index = data[f'scale{pooling_scale_num}_sample_index']

        for i in reversed(range(pooling_scale_num)):
            index = data[f'scale{i}_sample_index'][index.long()]

        pooling_visuals_tensor[index.long()] = pooling_scale_num

    return pooling_visuals_tensor


def ddp_setup(rank, num_gpus):

    if num_gpus > 1:

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        sys.stderr = open(f"{rank}.out", 'w')  # used by "tqdm"

        torch.distributed.init_process_group('nccl', rank=rank, world_size=num_gpus)
        wandb.init(project="lumen", config=wandb_config, group=f"DDP ({asctime()})")

    else:
        wandb.init(project="lumen", config=wandb_config)


def ddp_module(torch_module, rank):
    return DistributedDataParallel(torch_module, device_ids=[rank]) if torch.distributed.is_initialized() else torch_module


def ddp_rank_zero(fun, *args):

    if torch.distributed.is_initialized():
        fun(*args) if torch.distributed.get_rank() == 0 else None

    else:
        fun(*args)


def ddp_cleanup():

    wandb.finish()
    torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None

    sys.stderr.close() if torch.distributed.is_initialized() else None  # last executed statement


def ddp(fun, num_gpus):
    torch.multiprocessing.spawn(fun, args=(num_gpus,), nprocs=num_gpus, join=True) if num_gpus > 1 else fun(rank=0, num_gpus=num_gpus)


if __name__ == '__main__':
    ddp(main, args.num_gpus)
