import os
import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data, Batch
from dataset import HumanPointCloudDataset, analyze_dataset_info, compare_dataset_samples

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
    im = ax1.imshow(ect_np, cmap='viridis', aspect='auto', origin='lower')
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