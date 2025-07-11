from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from pc_vis_utils import display_part_pc



class HumanPointCloudDataset(Dataset):
    def __init__(self, vertices_hdf5_path, labels_hdf5_path, test_size=0.2, random_state=42, normalize=True, shuffle=False):
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
            self.dataset_names = sorted(list(common_dataset_names))
        
        # Train/test split at the model level
        train_indices, test_indices = train_test_split(
            range(len(self.dataset_names)), 
            test_size=test_size, 
            random_state=random_state,
            shuffle=shuffle
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


def analyze_dataset_info(dataset, sample_idx=0, train=True, detailed=True):
    """
    Analyze information about a specific dataset sample including point counts and labels
    
    Args:
        dataset: HumanPointCloudDataset instance
        sample_idx: Index of sample to analyze
        train: Whether to use training or test set
        detailed: Whether to show detailed label breakdown
    
    Returns:
        dict: Dictionary containing dataset analysis information
    """
    # Set the correct mode
    orig_mode = dataset.is_train
    dataset.set_mode(train)
    
    # Get the sample
    sample_data = dataset[sample_idx]
    
    # Reset the mode
    dataset.set_mode(orig_mode)
    
    # Extract points and labels
    points = sample_data.x.numpy()
    labels = sample_data.y.numpy()
    
    # Basic statistics
    total_points = len(points)
    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)
    
    # Label distribution
    label_counts = {}
    label_percentages = {}
    
    for label in unique_labels:
        count = np.sum(labels == label)
        percentage = (count / total_points) * 100
        label_counts[int(label)] = count
        label_percentages[int(label)] = percentage
    
    # Point cloud bounds
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    point_cloud_size = max_coords - min_coords
    
    # Create analysis dictionary
    analysis = {
        'sample_idx': sample_idx,
        'dataset_type': 'Training' if train else 'Test',
        'total_points': total_points,
        'num_unique_labels': num_unique_labels,
        'unique_labels': unique_labels.tolist(),
        'label_counts': label_counts,
        'label_percentages': label_percentages,
        'point_cloud_bounds': {
            'min': min_coords.tolist(),
            'max': max_coords.tolist(),
            'size': point_cloud_size.tolist()
        },
        'points_shape': points.shape,
        'labels_shape': labels.shape
    }
    
    # Print detailed information
    print("="*60)
    print(f"DATASET ANALYSIS - {analysis['dataset_type']} Sample {sample_idx}")
    print("="*60)
    
    print(f"Total Points: {total_points:,}")
    print(f"Point Cloud Shape: {points.shape}")
    print(f"Number of Unique Labels: {num_unique_labels}")
    print(f"Label Range: {min(unique_labels)} to {max(unique_labels)}")
    
    print(f"\nPoint Cloud Bounds:")
    print(f"  Min coordinates: [{min_coords[0]:.3f}, {min_coords[1]:.3f}, {min_coords[2]:.3f}]")
    print(f"  Max coordinates: [{max_coords[0]:.3f}, {max_coords[1]:.3f}, {max_coords[2]:.3f}]")
    print(f"  Size: [{point_cloud_size[0]:.3f}, {point_cloud_size[1]:.3f}, {point_cloud_size[2]:.3f}]")
    
    if detailed:
        print(f"\nLabel Distribution:")
        print("-" * 40)
        print(f"{'Label':<8} {'Count':<10} {'Percentage':<12}")
        print("-" * 40)
        
        # Sort labels for consistent output
        sorted_labels = sorted(unique_labels)
        for label in sorted_labels:
            count = label_counts[int(label)]
            percentage = label_percentages[int(label)]
            print(f"{label:<8} {count:<10,} {percentage:<12.2f}%")
        
        print("-" * 40)
        
        # Show top 5 most common labels
        sorted_by_count = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 Most Common Labels:")
        for i, (label, count) in enumerate(sorted_by_count[:5]):
            percentage = label_percentages[label]
            print(f"  {i+1}. Label {label}: {count:,} points ({percentage:.2f}%)")
        
        # Show labels with very few points (potential outliers)
        sparse_labels = [(label, count) for label, count in label_counts.items() if count < total_points * 0.01]  # Less than 1%
        if sparse_labels:
            print(f"\nSparse Labels (< 1% of points):")
            for label, count in sorted(sparse_labels, key=lambda x: x[1]):
                percentage = label_percentages[label]
                print(f"  Label {label}: {count:,} points ({percentage:.2f}%)")
    
    print("="*60)
    
    return analysis

def compare_dataset_samples(dataset, sample_indices=None, train=True, max_samples=5):
    """
    Compare multiple dataset samples to understand dataset variation
    
    Args:
        dataset: HumanPointCloudDataset instance
        sample_indices: List of sample indices to compare (if None, uses first max_samples)
        train: Whether to use training or test set
        max_samples: Maximum number of samples to compare
    
    Returns:
        list: List of analysis dictionaries for each sample
    """
    # Set the correct mode
    orig_mode = dataset.is_train
    dataset.set_mode(train)
    
    dataset_size = len(dataset)
    
    # Reset the mode
    dataset.set_mode(orig_mode)
    
    if sample_indices is None:
        sample_indices = list(range(min(max_samples, dataset_size)))
    else:
        sample_indices = sample_indices[:max_samples]  # Limit to max_samples
    
    print("="*70)
    print(f"COMPARING {len(sample_indices)} DATASET SAMPLES")
    print("="*70)
    
    analyses = []
    
    for idx in sample_indices:
        if idx >= dataset_size:
            print(f"Warning: Sample index {idx} exceeds dataset size {dataset_size}")
            continue
            
        analysis = analyze_dataset_info(dataset, sample_idx=idx, train=train, detailed=False)
        analyses.append(analysis)
        print()  # Add spacing between samples
    
    # Summary comparison
    if len(analyses) > 1:
        print("="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        total_points = [a['total_points'] for a in analyses]
        num_labels = [a['num_unique_labels'] for a in analyses]
        
        print(f"Point Count Statistics:")
        print(f"  Min: {min(total_points):,} points")
        print(f"  Max: {max(total_points):,} points") 
        print(f"  Mean: {np.mean(total_points):,.0f} points")
        print(f"  Std: {np.std(total_points):,.0f} points")
        
        print(f"\nLabel Count Statistics:")
        print(f"  Min unique labels: {min(num_labels)}")
        print(f"  Max unique labels: {max(num_labels)}")
        print(f"  Mean unique labels: {np.mean(num_labels):.1f}")
        
        # Find common labels across all samples
        all_labels_sets = [set(a['unique_labels']) for a in analyses]
        common_labels = set.intersection(*all_labels_sets) if all_labels_sets else set()
        all_labels_union = set.union(*all_labels_sets) if all_labels_sets else set()
        
        print(f"\nLabel Consistency:")
        print(f"  Common labels across all samples: {len(common_labels)} ({sorted(list(common_labels))})")
        print(f"  Total unique labels across all samples: {len(all_labels_union)}")
        print(f"  Label consistency: {len(common_labels)/len(all_labels_union)*100:.1f}%")
    
    print("="*70)
    
    return analyses
