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
from pc_vis_utils import display_part_pc

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# HDF5 Dataset class
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

def compute_ect_for_label(dataset, sample_idx, target_label, train=True, num_thetas=64, resolution=32, radius=1.0, scale=8):
    """
    Compute ECT features for points with a specific label in a dataset sample
    
    Args:
        dataset: HumanPointCloudDataset instance
        sample_idx: Index of sample in the dataset
        target_label: The specific label to filter points for
        train: Whether to use training or test set
        num_thetas: Number of directions for ECT
        resolution: Resolution of ECT grid
        radius: Radius for filtration
        scale: Scale parameter for ECT
    
    Returns:
        dict: Dictionary containing:
            - 'ect_features': ECT features tensor
            - 'num_points': Number of points with the target label
            - 'label': The target label
            - 'sample_idx': The sample index
            - 'filtered_points': The filtered point coordinates
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
    
    # Filter points for the target label
    label_mask = labels == target_label
    filtered_points = points[label_mask]
    
    if len(filtered_points) == 0:
        print(f"Warning: No points found with label {target_label} in sample {sample_idx}")
        return {
            'ect_features': None,
            'num_points': 0,
            'label': target_label,
            'sample_idx': sample_idx,
            'filtered_points': filtered_points
        }
    
    print(f"Computing ECT for label {target_label}: {len(filtered_points)} points")
    
    # Create new Data object with filtered points
    filtered_data = Data(x=torch.from_numpy(filtered_points.astype(np.float32)))
    
    try:
        # Compute ECT features for filtered points
        ect_features = compute_ect_features(
            filtered_data,
            num_thetas=num_thetas,
            resolution=resolution,
            radius=radius,
            scale=scale
        )
        
        return {
            'ect_features': ect_features,
            'num_points': len(filtered_points),
            'label': target_label,
            'sample_idx': sample_idx,
            'filtered_points': filtered_points
        }
        
    except Exception as e:
        print(f"ECT computation failed for label {target_label}: {e}")
        return {
            'ect_features': None,
            'num_points': len(filtered_points),
            'label': target_label,
            'sample_idx': sample_idx,
            'filtered_points': filtered_points
        }

def visualize_ect_features(ect_features, title="ECT Features", save_path=None, show_3d=True, show_stats=True):
    """
    Visualize ECT features with multiple visualization options
    
    Args:
        ect_features: ECT features tensor (can be on GPU or CPU)
        title: Title for the visualization
        save_path: Optional path to save the visualization
        show_3d: Whether to show 3D surface plot
        show_stats: Whether to show statistical information
    """
    # Convert to numpy and ensure 2D
    if isinstance(ect_features, torch.Tensor):
        ect_np = ect_features.detach().cpu().numpy()
    else:
        ect_np = ect_features
    
    # Handle different tensor shapes
    if ect_np.ndim == 1:
        # If 1D, try to reshape to square
        size = int(np.sqrt(len(ect_np)))
        if size * size == len(ect_np):
            ect_np = ect_np.reshape(size, size)
        else:
            # If not perfect square, reshape to reasonable 2D
            ect_np = ect_np.reshape(1, -1)
    elif ect_np.ndim > 2:
        # If more than 2D, squeeze or take first slice
        ect_np = ect_np.squeeze()
        if ect_np.ndim > 2:
            ect_np = ect_np[0]  # Take first slice
    
    # Create figure with subplots
    if show_3d and show_stats:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    elif show_3d or show_stats:
        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    else:
        fig = plt.figure(figsize=(8, 6))
        gs = fig.add_gridspec(1, 1)
    
    # 1. Main ECT heatmap
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(ect_np.T, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title(f'{title} - Heatmap')
    ax1.set_xlabel('Direction Index')
    ax1.set_ylabel('Filtration Level')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    subplot_idx = 1
    
    # 2. 3D surface plot (if requested and data is 2D)
    if show_3d and ect_np.ndim == 2 and min(ect_np.shape) > 1:
        ax2 = fig.add_subplot(gs[subplot_idx], projection='3d')
        x = np.arange(ect_np.shape[0])
        y = np.arange(ect_np.shape[1])
        X, Y = np.meshgrid(x, y)
        surf = ax2.plot_surface(X, Y, ect_np.T, cmap='viridis', alpha=0.8)
        ax2.set_title(f'{title} - 3D Surface')
        ax2.set_xlabel('Direction Index')
        ax2.set_ylabel('Filtration Level')
        ax2.set_zlabel('ECT Value')
        subplot_idx += 1
    
    # 3. Statistical summary (if requested)
    if show_stats:
        ax3 = fig.add_subplot(gs[subplot_idx])
        
        # Plot some statistics
        if ect_np.ndim == 2:
            # Row-wise and column-wise statistics
            row_means = np.mean(ect_np, axis=1)
            col_means = np.mean(ect_np, axis=0)
            
            ax3_twin = ax3.twinx()
            
            line1 = ax3.plot(row_means, 'b-', label='Direction Means', linewidth=2)
            line2 = ax3_twin.plot(col_means, 'r-', label='Filtration Means', linewidth=2)
            
            ax3.set_xlabel('Index')
            ax3.set_ylabel('Direction Means', color='b')
            ax3_twin.set_ylabel('Filtration Means', color='r')
            ax3.tick_params(axis='y', labelcolor='b')
            ax3_twin.tick_params(axis='y', labelcolor='r')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right')
            
        else:
            # For 1D data, just plot the values
            ax3.plot(ect_np.flatten(), 'g-', linewidth=2)
            ax3.set_xlabel('Index')
            ax3.set_ylabel('ECT Value')
        
        ax3.set_title(f'{title} - Statistics')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print some statistics
    print(f"\nECT Features Statistics for '{title}':")
    print(f"Shape: {ect_np.shape}")
    print(f"Min: {ect_np.min():.4f}")
    print(f"Max: {ect_np.max():.4f}")
    print(f"Mean: {ect_np.mean():.4f}")
    print(f"Std: {ect_np.std():.4f}")
    print(f"Non-zero elements: {np.count_nonzero(ect_np)}/{ect_np.size}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def visualize_ect_comparison(ect_features_list, titles=None, save_path=None):
    """
    Compare multiple ECT features side by side
    
    Args:
        ect_features_list: List of ECT features tensors
        titles: List of titles for each ECT feature set
        save_path: Optional path to save the comparison
    """
    n_features = len(ect_features_list)
    
    if titles is None:
        titles = [f"ECT {i+1}" for i in range(n_features)]
    
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 4))
    if n_features == 1:
        axes = [axes]
    
    for i, (ect_features, title) in enumerate(zip(ect_features_list, titles)):
        # Convert to numpy
        if isinstance(ect_features, torch.Tensor):
            ect_np = ect_features.detach().cpu().numpy()
        else:
            ect_np = ect_features
        
        # Handle tensor shapes
        if ect_np.ndim == 1:
            size = int(np.sqrt(len(ect_np)))
            if size * size == len(ect_np):
                ect_np = ect_np.reshape(size, size)
            else:
                ect_np = ect_np.reshape(1, -1)
        elif ect_np.ndim > 2:
            ect_np = ect_np.squeeze()
            if ect_np.ndim > 2:
                ect_np = ect_np[0]
        
        # Plot heatmap
        im = axes[i].imshow(ect_np.T, cmap='viridis', aspect='auto', origin='lower')
        axes[i].set_title(title)
        axes[i].set_xlabel('Direction Index')
        if i == 0:
            axes[i].set_ylabel('Filtration Level')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

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

def run_ect_pipeline(dataset):

    print("="*50)
    print("RUNNING ECT PIPELINE")
    print("="*50)
    
    if len(dataset) > 0:
        # Analyze dataset information first
        print("\nAnalyzing dataset information...")
        analyze_dataset_info(dataset, sample_idx=22, train=True, detailed=False)
        
        # Compare multiple samples if available
        # if len(dataset) > 1:
        #     print(f"\nComparing multiple dataset samples...")
        #     compare_dataset_samples(dataset, sample_indices=None, train=True, max_samples=3)
        
        # Test ECT computation on first sample
        # test_ect_computation(dataset, sample_idx=22, train=True)
        
        # # Test ECT computation for specific label
        print("\nTesting ECT computation for specific label...")
        result = compute_ect_for_label(dataset, sample_idx=22, target_label=2, train=True)
        if result['ect_features'] is not None:
            print(f"ECT features for label {result['label']}: shape {result['ect_features'].shape}")
            print(f"Number of points with label {result['label']}: {result['num_points']}")
            
            # Visualize ECT features
            print("\nVisualizing ECT features...")
            visualize_ect_features(
                result['ect_features'], 
                title=f"ECT for Label {result['label']} ({result['num_points']} points)",
                show_3d=True,
                show_stats=True
            )

            point_labels = np.full(len(result['filtered_points']), result['label'])
            display_part_pc(result['filtered_points'], point_labels)

            
        
        # Compare ECT features for different labels
        # print("\nComparing ECT features for different labels...")
        # ect_comparison = []
        # comparison_titles = []
        
        # for label in [0, 1, 2]:  # Compare first 3 labels
        #     label_result = compute_ect_for_label(dataset, sample_idx=0, target_label=label, train=True)
        #     if label_result['ect_features'] is not None and label_result['num_points'] > 10:  # Only include if enough points
        #         ect_comparison.append(label_result['ect_features'])
        #         comparison_titles.append(f"Label {label} ({label_result['num_points']} pts)")
        
        # if len(ect_comparison) > 1:
        #     visualize_ect_comparison(ect_comparison, comparison_titles)
        
        # You can add more ECT processing steps here later
        
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
    run_ect_pipeline(dataset)
    
    # Visualize a specific sample
    dataset.visualize_sample(22, train=True, with_normals=False, with_default_view=True)
    
    # # Visualize a sample if available
    # if len(dataset) > 0:
    #     # Visualize the first sample
    #     dataset.visualize_sample(0, train=True)