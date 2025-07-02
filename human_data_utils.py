import os
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HumanPointCloudDataset(Dataset):
    def __init__(self, hdf5_path, transform=None, pre_transform=None, pre_filter=None):
        """
        Dataset for human point clouds with body part labels
        
        Args:
            hdf5_path: Path to the HDF5 file with human point clouds
        """
        super().__init__(None, transform, pre_transform, pre_filter)
        self.hdf5_path = hdf5_path
        
        # Get list of models from the HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            self.model_ids = list(f.keys())
        
        print(f"Loaded {len(self.model_ids)} human models from {hdf5_path}")
    
    def len(self):
        return len(self.model_ids)
    
    def get(self, idx):
        """Get a model by index"""
        model_id = self.model_ids[idx]
        
        # Load vertices and labels
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[model_id]
            vertices = np.array(group['sampled_vertices'][:], dtype=np.float32)
            labels = np.array(group['sampled_labels'][:], dtype=np.int64)
        
        # Normalize coordinates to [-1, 1]
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        vertices = 2 * (vertices - vertices_min) / (vertices_max - vertices_min) - 1
        
        # Convert to PyTorch tensors
        vertices_tensor = torch.from_numpy(vertices)
        labels_tensor = torch.from_numpy(labels)
        
        # Create PyTorch Geometric Data object
        data = Data(x=vertices_tensor, y=labels_tensor, model_id=model_id)
        
        return data
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset"""
        all_labels = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for model_id in self.model_ids:
                labels = np.array(f[model_id]['sampled_labels'][:])
                all_labels.append(labels)
        
        all_labels = np.concatenate(all_labels)
        return len(np.unique(all_labels))
    
    def visualize(self, idx, with_labels=True):
        """Visualize a sample from the dataset"""
        data = self.get(idx)
        vertices = data.x.numpy()
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if with_labels:
            labels = data.y.numpy()
            scatter = ax.scatter(
                vertices[:, 0], 
                vertices[:, 1], 
                vertices[:, 2], 
                c=labels, 
                cmap='tab20', 
                s=5, 
                alpha=0.8
            )
            plt.colorbar(scatter, ax=ax, label='Body Part Labels')
            title = f'Human Point Cloud - Model {data.model_id} with Labels'
        else:
            ax.scatter(
                vertices[:, 0], 
                vertices[:, 1], 
                vertices[:, 2], 
                s=5, 
                alpha=0.8
            )
            title = f'Human Point Cloud - Model {data.model_id}'
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.max([
            np.ptp(vertices[:, 0]), 
            np.ptp(vertices[:, 1]), 
            np.ptp(vertices[:, 2])
        ])
        mid_x = np.mean([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
        mid_y = np.mean([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
        mid_z = np.mean([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    data_dir = './data'
    hdf5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.hdf5') or f.endswith('.h5')]
    
    if hdf5_files:
        dataset = HumanPointCloudDataset(hdf5_files[0])
        
        # Print dataset info
        print(f"Dataset contains {len(dataset)} human models")
        print(f"Number of body part classes: {dataset.get_num_classes()}")
        
        # Visualize a few samples
        for i in range(min(3, len(dataset))):
            dataset.visualize(i)
    else:
        print("No HDF5 files found in the data directory.")
