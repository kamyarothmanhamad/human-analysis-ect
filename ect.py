import os
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split

# Import DECT components
import sys
sys.path.append(os.path.abspath('../dect'))  # Add the DECT package to path
from dect.directions import generate_uniform_directions
from dect.nn import ECTLayer, ECTConfig

# Import visualization functions
from vis_dataset import display_part_pc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# HDF5 Dataset class
class HumanPointCloudDataset(Dataset):
    def __init__(self, vertices_hdf5_path, labels_hdf5_path, test_size=0.2, random_state=42, normalize=True):
        """
        Dataset for human point clouds with body part labels from separate HDF5 files
        
        Args:
            vertices_hdf5_path: Path to the HDF5 file containing vertices/points
            labels_hdf5_path: Path to the HDF5 file containing labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for train/test split
            normalize: Whether to normalize point coordinates
        """
        self.vertices_hdf5_path = vertices_hdf5_path
        self.labels_hdf5_path = labels_hdf5_path
        self.normalize = normalize
        self.dataset_names = []
        self._cache = {}  # Cache for loaded samples
        
        # Only load dataset names and metadata, not the actual data
        with h5py.File(vertices_hdf5_path, 'r') as vertices_file, h5py.File(labels_hdf5_path, 'r') as labels_file:
            # Get all dataset names from vertices file and labels file
            vertices_dataset_names = list(vertices_file.keys())
            labels_dataset_names = list(labels_file.keys())
            
            # Find common dataset names between both files
            common_dataset_names = set(vertices_dataset_names) & set(labels_dataset_names)
            
            if not common_dataset_names:
                raise ValueError("No common dataset names found between vertices and labels files")
            
            # Store only the dataset names for lazy loading
            self.dataset_names = list(common_dataset_names)
        
        # Train/test split at the model level
        train_indices, test_indices = train_test_split(
            range(len(self.dataset_names)), 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Set the indices based on the split
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.is_train = True  # Default to training set
        
    def _load_sample(self, dataset_name):
        """Load a single sample from HDF5 files"""
        if dataset_name in self._cache:
            return self._cache[dataset_name]
        
        with h5py.File(self.vertices_hdf5_path, 'r') as vertices_file, h5py.File(self.labels_hdf5_path, 'r') as labels_file:
            # Load vertices and labels
            vertices = np.array(vertices_file[dataset_name][:])
            labels = np.array(labels_file[dataset_name][:])
            
            # Verify that vertices and labels have the same number of points
            if len(vertices) != len(labels):
                raise ValueError(f"Dataset '{dataset_name}' - vertices ({len(vertices)}) and labels ({len(labels)}) have different lengths")
            
            sample = {
                'model_id': dataset_name,
                'vertices': vertices,
                'labels': labels
            }
            
            # Cache the sample
            self._cache[dataset_name] = sample
            return sample
        
    def set_mode(self, train=True):
        """Switch between training and test sets"""
        self.is_train = train
    
    def __len__(self):
        """Return the number of samples in the current set"""
        return len(self.train_indices if self.is_train else self.test_indices)
    
    def __getitem__(self, idx):
        """Get a sample by index"""
        # Map the index to the corresponding dataset name
        dataset_idx = self.train_indices[idx] if self.is_train else self.test_indices[idx]
        dataset_name = self.dataset_names[dataset_idx]
        
        # Load the sample (from cache if available)
        sample = self._load_sample(dataset_name)
        
        # Get vertices and labels
        vertices = sample['vertices'].astype(np.float32)
        labels = sample['labels'].astype(np.int64)
        
        # Normalize coordinates if specified
        if self.normalize:
            # Center by subtracting mean
            center = vertices.mean(axis=0)
            vertices = vertices - center
            
            # Scale to unit norm (unit sphere)
            max_norm = np.sqrt((vertices**2).sum(axis=1).max())
            if max_norm > 0:
                vertices = vertices / max_norm
        
        # Convert to PyTorch tensors
        vertices_tensor = torch.from_numpy(vertices)
        labels_tensor = torch.from_numpy(labels)
        
        # Create PyTorch Geometric Data object
        data = Data(x=vertices_tensor, y=labels_tensor)
        
        return data
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset"""
        all_labels = []
        
        # Load all samples to get unique labels
        for dataset_name in self.dataset_names:
            sample = self._load_sample(dataset_name)
            all_labels.append(sample['labels'])
        
        all_labels = np.concatenate(all_labels)
        return len(np.unique(all_labels))

    def visualize_sample(self, idx, train=True, with_normals=False, with_default_view=True):
        """Visualize a sample from the dataset using Open3D"""
        # Set the correct mode
        orig_mode = self.is_train
        self.set_mode(train)
        
        # Get the sample
        data = self[idx]
        vertices = data.x.numpy()
        labels = data.y.numpy()
        
        # Reset the mode
        self.set_mode(orig_mode)
        
        # Use Open3D visualization
        print(f"Visualizing {'Training' if train else 'Test'} Sample {idx}")
        print(f"Points: {len(vertices)}, Unique labels: {len(np.unique(labels))}")
        
        display_part_pc(vertices, labels, with_normals=with_normals, with_default_view=with_default_view)


# Custom collate function for batching PyG Data objects
def collate_fn(batch):
    return Batch.from_data_list(batch)

def compute_ect_features(data, num_thetas=64, resolution=32, radius=1.0, scale=8):
    """
    Compute ECT features for a given sample
    
    Args:
        data: PyG Data object with point cloud
        num_thetas: Number of directions for ECT
        resolution: Resolution of ECT grid
        radius: Radius for filtration
        scale: Scale parameter for ECT
    
    Returns:
        ECT features tensor
    """
    # Generate uniform directions
    directions = generate_uniform_directions(
        num_thetas=num_thetas,
        d=3,  # 3D space
        seed=42,
        device=device
    )
    
    # Create ECT layer
    ect_layer = ECTLayer(
        ECTConfig(
            ect_type="points",
            resolution=resolution,
            scale=scale,
            radius=radius,
            normalized=False,  # Set to False to avoid dimension error
            fixed=True  # Use fixed directions
        ),
        v=directions
    ).to(device)
    
    # Create batch and move to device
    batch = Batch.from_data_list([data]).to(device)
    
    # Compute ECT features
    ect_features = ect_layer(batch)
    
    # Manual normalization if needed
    if ect_features.dim() >= 2 and ect_features.max() != 0:
        max_val = ect_features.max()
        ect_features = ect_features / max_val
    
    return ect_features

def test_ect_computation(dataset, sample_idx=0, train=True):
    """
    Test ECT computation on a sample from the dataset
    
    Args:
        dataset: HumanPointCloudDataset instance
        sample_idx: Index of sample to test
        train: Whether to use training or test set
    """
    # Set the correct mode
    orig_mode = dataset.is_train
    dataset.set_mode(train)
    
    # Get the sample
    sample_data = dataset[sample_idx]
    
    # Reset the mode
    dataset.set_mode(orig_mode)
    
    print(f"Testing ECT computation on {'Training' if train else 'Test'} Sample {sample_idx}")
    print(f"Points: {sample_data.x.shape[0]}, Features: {sample_data.x.shape[1]}")
    
    try:
        ect_features = compute_ect_features(
            sample_data, 
            num_thetas=24, 
            resolution=16, 
            radius=1.0, 
            scale=8
        )
        print(f"ECT features shape: {ect_features.shape}")
        print(f"ECT tensor dimensions: {ect_features.dim()}")
        print("ECT computation succeeded!")
        
    except Exception as e:
        print(f"ECT computation failed: {e}")

def run_ect_pipeline(dataset):
    """
    Run the complete ECT processing pipeline
    
    Args:
        dataset: HumanPointCloudDataset instance
    """
    print("="*50)
    print("RUNNING ECT PIPELINE")
    print("="*50)
    
    if len(dataset) > 0:
        # Test ECT computation on first sample
        test_ect_computation(dataset, sample_idx=0, train=True)
        
        # You can add more ECT processing steps here later
        # For example: batch processing, feature extraction, etc.
        
    print("\n" + "="*50)
    print("ECT PIPELINE COMPLETED")
    print("="*50)

if __name__ == "__main__":
    # Path to your HDF5 files - replace with your actual file paths
    vertices_path = "./data/reoriented_vertices.hdf5"
    labels_path = "./data/cihp_vertex_labels.hdf5"
    
    # Create the dataset
    dataset = HumanPointCloudDataset(
        vertices_hdf5_path=vertices_path,
        labels_hdf5_path=labels_path,
        normalize=True
    )
    
    # Print basic information
    print(f"Dataset loaded: {len(dataset)} training samples")
    # print(f"Number of classes: {dataset.get_num_classes()}")
    
    # Run ECT pipeline
    # run_ect_pipeline(dataset)
    
    # # Visualize a sample if available
    # if len(dataset) > 0:
    #     # Visualize the first sample
    #     dataset.visualize_sample(0, train=True)