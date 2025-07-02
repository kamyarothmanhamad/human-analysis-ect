import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def explore_hdf5_structure(file_path):
    """
    Explore and print the structure of an HDF5 file in a simple, hierarchical format
    """
    print(f"\nExploring HDF5 file: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Get top-level keys
            keys = list(f.keys())
            print(f"Top-level keys: {keys}")
            
            # Explore the first key in detail
            if keys:
                first_key = keys[0]
                print(f"\nExploring first key: '{first_key}'")
                
                data = f[first_key]
                if isinstance(data, h5py.Dataset):
                    # This is a dataset (contains actual data)
                    print(f"  Type: Dataset")
                    print(f"  Shape: {data.shape}")
                    print(f"  Data type: {data.dtype}")
                    
                    # Show a small sample of the data
                    sample_data = data[0] if data.shape[0] > 0 else None
                    if sample_data is not None:
                        print(f"  First item shape: {sample_data.shape if hasattr(sample_data, 'shape') else 'Scalar'}")
                        print(f"  Sample data: {sample_data[:5] if hasattr(sample_data, '__len__') else sample_data}")
                    
                    # Try to visualize if it looks like point data
                    if len(data.shape) >= 2 and data.shape[1] >= 3:
                        visualize_sample(data[:])
                    
                elif isinstance(data, h5py.Group):
                    # This is a group (contains other groups or datasets)
                    subkeys = list(data.keys())
                    print(f"  Type: Group")
                    print(f"  Subkeys: {subkeys}")
                    
                    # For each subkey, print info
                    for subkey in subkeys:
                        subdata = data[subkey]
                        print(f"\n  Exploring subkey: '{subkey}'")
                        if isinstance(subdata, h5py.Dataset):
                            print(f"    Type: Dataset")
                            print(f"    Shape: {subdata.shape}")
                            print(f"    Data type: {subdata.dtype}")
                            
                            # Show a small sample of the subdata if it's not too big
                            if len(subdata.shape) > 0 and subdata.shape[0] > 0:
                                sample = subdata[0] if subdata.shape[0] > 0 else None
                                if sample is not None:
                                    sample_str = str(sample)
                                    if len(sample_str) > 100:
                                        sample_str = sample_str[:100] + "..."
                                    print(f"    First item: {sample_str}")
                    
                    # If there are vertices and labels, try to visualize them together
                    if 'sampled_vertices' in subkeys and 'sampled_labels' in subkeys:
                        try:
                            vertices = data['sampled_vertices'][:]
                            labels = data['sampled_labels'][:]
                            print(f"\nLoaded vertices shape: {vertices.shape}")
                            print(f"Loaded labels shape: {labels.shape}")
                            
                            # Visualize if the vertices look like 3D points
                            if len(vertices.shape) >= 2 and vertices.shape[1] >= 3:
                                visualize_sample(vertices, labels)
                            else:
                                print("Vertices data doesn't look like 3D points, skipping visualization")
                        except Exception as e:
                            print(f"Error loading vertices and labels: {e}")
                
    except Exception as e:
        print(f"Error exploring HDF5 file: {e}")

def visualize_sample(points, labels=None, sample_size=1000):
    """Visualize a small sample of 3D points with optional labels"""
    # Sample points if there are too many
    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[indices]
        if labels is not None:
            sample_labels = labels[indices]
    else:
        sample_points = points
        sample_labels = labels
    
    print(f"\nVisualizing sample of {len(sample_points)} points from data")
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points, coloring by label if available
    if labels is not None:
        scatter = ax.scatter(
            sample_points[:, 0], 
            sample_points[:, 1], 
            sample_points[:, 2], 
            c=sample_labels, 
            cmap='tab20', 
            s=10, 
            alpha=0.8
        )
        plt.colorbar(scatter, ax=ax, label='Labels')
        title = '3D Point Cloud Sample with Labels'
    else:
        ax.scatter(
            sample_points[:, 0], 
            sample_points[:, 1], 
            sample_points[:, 2], 
            s=10, 
            alpha=0.8
        )
        title = '3D Point Cloud Sample'
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.max([
        np.ptp(sample_points[:, 0]), 
        np.ptp(sample_points[:, 1]), 
        np.ptp(sample_points[:, 2])
    ])
    mid_x = np.mean([np.min(sample_points[:, 0]), np.max(sample_points[:, 0])])
    mid_y = np.mean([np.min(sample_points[:, 1]), np.max(sample_points[:, 1])])
    mid_z = np.mean([np.min(sample_points[:, 2]), np.max(sample_points[:, 2])])
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()

# Try to find HDF5 files in the data directory
data_dir = './data'
print(f"Looking for HDF5 files in: {os.path.abspath(data_dir)}")

try:
    # List available HDF5 files
    hdf5_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.hdf5') or f.endswith('.h5')]
    
    if hdf5_files:
        print(f"Found {len(hdf5_files)} HDF5 files:")
        for i, file in enumerate(hdf5_files):
            print(f"{i+1}. {os.path.basename(file)}")
        
        # Explore the first file
        first_file = hdf5_files[0]
        print(f"\nAutomatically selecting first file: {os.path.basename(first_file)}")
        explore_hdf5_structure(first_file)
    else:
        print("No HDF5 files found in the data directory.")

except FileNotFoundError:
    print(f"Directory not found: {data_dir}")
except Exception as e:
    print(f"Error: {e}")
