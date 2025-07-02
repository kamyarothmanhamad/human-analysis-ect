import h5py
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

def load_and_visualize_first_model(hdf5_path):
    """
    Load the first model from an HDF5 file and visualize it with trimesh
    """
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Get the first model name
        model_names = list(f.keys())
        if not model_names:
            print("No models found in the HDF5 file")
            return
        
        first_model_name = model_names[88]
        print(f"Loading model: {first_model_name}")
        
        # Access the vertices and labels for the first model
        model_group = f[first_model_name]
        vertices = np.array(model_group['sampled_vertices'])
        labels = np.array(model_group['sampled_labels'])
        
        # Print some information about the loaded data
        print(f"Vertices shape: {vertices.shape}")
        print(f"Labels shape: {labels.shape}")
        unique_labels = np.unique(labels)
        print(f"Number of unique labels: {len(unique_labels)}")
        print(f"Label values: {unique_labels}")
        
        # Define a color map with distinct colors for better visibility
        # Using a high-contrast color palette for better distinction between body parts
        color_palette = {
            2: [0.94, 0.12, 0.12],    # Bright Red
            5: [0.13, 0.55, 0.13],    # Forest Green
            9: [0.11, 0.27, 0.81],    # Royal Blue
            10: [0.99, 0.75, 0.05],   # Golden Yellow
            13: [0.58, 0.0, 0.83],    # Vivid Purple
            14: [0.0, 0.74, 0.83],    # Turquoise
            15: [0.95, 0.33, 0.0],    # Burnt Orange
            18: [0.74, 0.18, 0.39],   # Crimson
            19: [0.37, 0.62, 0.22],   # Olive Green
        }
        
        # Generate colors for all unique labels not in the palette
        print("\nGenerating colors for labels not in predefined palette:")
        np.random.seed(42)  # For reproducible colors
        
        for label in unique_labels:
            if label not in color_palette:
                print(f"Generating color for label {label}")
                # Generate distinct colors with good separation
                hue = (hash(int(label)) % 10) / 10.0  # Use hash for deterministic but distributed hues
                saturation = 0.8  # High saturation for vivid colors
                value = 0.9  # High value/brightness for visibility
                rgb_color = hsv_to_rgb([hue, saturation, value])
                color_palette[label] = rgb_color
        
        # Map labels to colors and check for any missing mappings
        colors = []
        missing_labels = set()
        for label in labels:
            if label in color_palette:
                colors.append(color_palette[label])
            else:
                colors.append([1, 0, 1])  # Magenta for missing colors (should not happen)
                missing_labels.add(label)
        
        if missing_labels:
            print(f"Warning: No colors assigned for labels: {missing_labels}")
        
        colors = np.array(colors)
        
        # Create a point cloud visualization
        cloud = trimesh.PointCloud(vertices=vertices, colors=colors * 255)
        
        # Visualize the point cloud
        scene = trimesh.Scene(cloud)
        
        # Print color legend for reference
        print("\nColor Legend:")
        for label in sorted(unique_labels):
            if label in color_palette:
                color = color_palette[label]
                print(f"Label {label}: RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})")
        
        scene.show()

if __name__ == "__main__":
    # Path to the HDF5 file
    hdf5_path = "data/cihp_sampled_pcs_data.hdf5"
    
    # Load and visualize the first model
    load_and_visualize_first_model(hdf5_path)
