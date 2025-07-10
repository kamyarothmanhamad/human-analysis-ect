"""
HDF5 Utility Module - Collection of functions for working with HDF5 files
"""
import h5py
import numpy as np


def inspect_h5_file(filepath):
    """Prints the structure (groups/datasets) of an HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        print(f"Inspecting: {filepath}")
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {node.shape}, dtype: {node.dtype}")
            elif isinstance(node, h5py.Group):
                print(f"Group: {name}")
        f.visititems(visitor)


def read_dataset(filepath, dataset_path):
    """Reads and returns a dataset from a given path in the HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        return f[dataset_path][:]
    

def list_datasets(filepath):
    """Returns a list of dataset paths in the HDF5 file."""
    datasets = []
    with h5py.File(filepath, 'r') as f:
        f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
    return datasets


def get_dataset_info(filepath, dataset_path):
    """Returns shape and dtype info of a dataset."""
    with h5py.File(filepath, 'r') as f:
        dset = f[dataset_path]
        return {'shape': dset.shape, 'dtype': dset.dtype}
