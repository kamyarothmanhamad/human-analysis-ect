import sys
import os
import open3d as o3d
import numpy as np
import hdf5_utils
import math
# sys.path.append('/home/kamyar/research/custom_codes/utils/')
# from vis import pc_vis


def get_part_color_map(num_vals: int) -> dict:
    keys = list(range(-1, num_vals+1))
    color_d = {}
    biases, gain, power = [80, 160, 240], 2, 3
    for key in keys:
        r = int(math.pow((key+biases[0])*gain, power)) % 255.0
        g = int(math.pow((key+biases[1])*gain, power)) % 255.0
        b = int(math.pow((key+biases[2])*gain, power)) % 255.0
        color_d[key] = np.array([r, g, b], dtype=np.uint8)
    return color_d

def parts_to_colors(parts: np.ndarray, color_map: dict = None) -> np.ndarray:
    part_colors = np.zeros((parts.shape[0], 3))
    unique_parts = np.unique(parts)
    if color_map is None:
        color_map = get_part_color_map(np.max(unique_parts)+1)
    for unique_part in unique_parts:
        part_colors[parts == unique_part] = color_map[unique_part]
    return part_colors

def display_part_pc(pc: np.ndarray, parts: np.ndarray,
                    with_normals: bool = False, with_default_view: bool = True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    colors = parts_to_colors(parts, None)
    colors /= 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if with_normals:
        pcd.estimate_normals()
    if with_default_view:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        view_control.set_front([0, 0, -1])
        view_control.set_zoom(0.8)
        vis.run()
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd])



if __name__ == "__main__":
    file_path_v = '/media/kamyar/669AE7069AE6D19B/Users/othma/Downloads/reoriented_vertices.hdf5'
    file_path_l = '/media/kamyar/669AE7069AE6D19B/Users/othma/Downloads/cihp_vertex_labels.hdf5'

    dataset_path_v = 'THuman2.1_Release_1201-1500_1293'
    dataset_path_l = 'THuman2.1_Release_1201-1500_1293'
    
    
    # Read the datasets
    ver = hdf5_utils.read_dataset(file_path_v, dataset_path_v)
    lab = hdf5_utils.read_dataset(file_path_l, dataset_path_l)
    
    
    # List datasets in the file
    # datasets = hdf5_utils.list_datasets(file_path)
    # print(datasets)
    
    
    # Display point cloud with parts
    display_part_pc(ver, lab)
