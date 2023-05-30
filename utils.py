import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np

import torch
from torch_scatter import scatter
from prettytable import PrettyTable


class VTUWriter():
    def __init__(self):
        self.vtu_writer = vtk.vtkXMLUnstructuredGridWriter()

    def __call__(self, path_to_file, data):

        self.vtu_writer.SetFileName(path_to_file)
        self.vtu_writer.SetInputData(self.pyg_to_vtk(data))

        self.vtu_writer.Update()

    def pyg_to_vtk(self, data):
        vtk_unstructured_grid = vtk.vtkUnstructuredGrid()

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(data.pos.numpy()))

        vtk_cell_array = vtk.vtkCellArray()
        vtk_cell_array.SetCells(
            data['tets' if 'tets' in data else 'face'].shape[-1],
            numpy_to_vtkIdTypeArray(self.serialise_simplices(data['tets' if 'tets' in data else 'face']))
        )

        vtk_unstructured_grid.SetPoints(vtk_points)
        vtk_unstructured_grid.SetCells(vtk.VTK_TETRA if 'tets' in data else vtk.VTK_POLYGON, vtk_cell_array)

        vtk_unstructured_grid = self.add_point_data(vtk_unstructured_grid, data)

        return vtk_unstructured_grid

    @staticmethod
    def serialise_simplices(simplices):
        simplices = simplices.t().numpy()  # (4, N) to (N, 4)

        simplices = np.concatenate((
            np.full(simplices.shape[0], simplices.shape[1])[:, None],
            simplices
        ), axis=-1)

        return simplices.ravel()

    def add_point_data(self, vtk_unstructured_grid, data):
        for key, value in {**self.parse_point_data(data), **self.parse_point_indices(data), **self.lift_data_to_volume_surface(data)}.items():

            array = numpy_to_vtk(value)
            array.SetName(key)

            vtk_unstructured_grid.GetPointData().AddArray(array)

        return vtk_unstructured_grid

    @staticmethod
    def parse_point_data(data):
        return {key: value for key, value in data if value.size(0) == data.num_nodes and key != 'pos' and value.dim() <= 2}

    @staticmethod
    def parse_point_indices(data):
        point_mask_dict = {}

        for key, value in data:
            if "_index" in key and key != 'edge_index':

                point_mask = np.zeros(data.num_nodes, dtype='f4')  # integer data types mess with VTK
                point_mask[value] = 1.

                point_mask_dict[key.replace("_index", "")] = point_mask

        return point_mask_dict

    @staticmethod
    def lift_data_to_volume_surface(data):
        dummy_volume_data_dict = {}

        for index_key, index in data:
            if "_index" in index_key:

                for key, value in data:
                    if value.size(0) == index.size(0) and key != index_key:

                        dummy_volume_allocation = np.zeros((data.num_nodes, *value.shape[1:]))
                        dummy_volume_allocation[index] = value

                        dummy_volume_data_dict[key] = dummy_volume_allocation

        return dummy_volume_data_dict


class AccuracyAnalysis():
    def __init__(self):

        self.values_dict = {
            'ground_truth': [],
            'prediction': [],
            'scatter_idx': []
        }

    def append_values(self, value_dict):

        for key, value in value_dict.items():
            if key in self.values_dict:

                if key == 'scatter_idx':
                    self.values_dict[key].append(value.expand(value_dict['ground_truth'].size(0)))

                else:
                    self.values_dict[key].append(value)

    def lists_to_tensors(self):
        self.values_dict = {key: torch.cat(value, dim=0) for key, value in self.values_dict.items()}

    def get_nmae(self):

        mae = scatter(
            torch.linalg.norm(self.values_dict['ground_truth'] - self.values_dict['prediction'], dim=-1),
            self.values_dict['scatter_idx'],
            dim=0,
            reduce='mean'
        )

        return mae / torch.max(torch.linalg.norm(self.values_dict['ground_truth'], dim=-1))

    def get_approximation_error(self):

        approximation_error = torch.sqrt(scatter(
            torch.linalg.norm(self.values_dict['ground_truth'] - self.values_dict['prediction'], dim=-1) ** 2,
            self.values_dict['scatter_idx'],
            dim=0,
            reduce='sum'
        ) / scatter(
            torch.linalg.norm(self.values_dict['ground_truth'], dim=-1) ** 2,
            self.values_dict['scatter_idx'],
            dim=0,
            reduce='sum'
        ))

        return approximation_error

    def get_mean_cosine_similarity(self):

        cosine_similarity = torch.nn.CosineSimilarity(dim=-1).forward(
            self.values_dict['ground_truth'],
            self.values_dict['prediction']
        )

        mean_cosine_similarity = scatter(
            cosine_similarity,
            self.values_dict['scatter_idx'],
            dim=0,
            reduce='mean'
        )

        return mean_cosine_similarity

    def accuracy_table(self):

        self.lists_to_tensors()

        nmae = self.get_nmae()
        approximation_error = self.get_approximation_error()
        mean_cosine_similarity = self.get_mean_cosine_similarity()

        table = PrettyTable(["Metric", "Mean", "Standard Deviation"])

        table.add_row([
            "NMAE",
            "{0:.1%}".format(torch.mean(nmae).item()),
            "{0:.1%}".format(torch.std(nmae).item())
        ])

        table.add_row([
            "Approximation Error",
            "{0:.1%}".format(torch.mean(approximation_error).item()),
            "{0:.1%}".format(torch.std(approximation_error).item())
        ])

        table.add_row([
            "Mean Cosine Similarity",
            "{:.2f}".format(torch.mean(mean_cosine_similarity).item()),
            "{:.2f}".format(torch.std(mean_cosine_similarity).item())
        ])

        return table
