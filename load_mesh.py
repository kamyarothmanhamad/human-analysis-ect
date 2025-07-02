import trimesh
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add the ECT package to path
sys.path.append(str(Path(__file__).parent.parent))

# Import ECT modules
from dect.directions import generate_uniform_directions
from dect.ect import compute_ect
from dect.ect_fn import indicator
from dect.nn import ECTLayer, ECTConfig
from torch_geometric.data import Data, Batch

def visualize_mesh(mesh):
    try:
        mesh.show()
        return True
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return False

def mesh_to_point_cloud(mesh, num_points=1000):
    """
    Convert a mesh to a point cloud by sampling points on the surface.
    
    Args:
        mesh: trimesh.Trimesh object
        num_points: Number of points to sample
        
    Returns:
        torch.Tensor: Point cloud tensor of shape [num_points, 3]
    """
    # Sample points on the mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Convert to torch tensor
    return torch.tensor(points, dtype=torch.float32)

def normalize_point_cloud(point_cloud):
    """
    Normalize point cloud to be centered at origin and have unit scale.
    
    Args:
        point_cloud: torch.Tensor of shape [N, D]
        
    Returns:
        torch.Tensor: Normalized point cloud
    """
    # Center
    centroid = point_cloud.mean(dim=0)
    centered = point_cloud - centroid
    
    # Scale
    scale = torch.max(torch.norm(centered, dim=1))
    normalized = centered / scale
    
    return normalized

def process_point_cloud_with_ect(point_cloud, num_directions=100, resolution=30, radius=1.0, scale=10.0, device="cpu"):
    """
    Process a point cloud using the Euler Characteristic Transform (ECT).
    
    Args:
        point_cloud: torch.Tensor of shape [N, D]
        num_directions: Number of directions to use
        resolution: Resolution for ECT computation
        radius: Radius for ECT computation
        scale: Scale factor for ECT computation
        device: Device to use for computation ('cpu', 'cuda', 'mps')
        
    Returns:
        torch.Tensor: ECT of the point cloud
    """
    # Move to specified device
    if torch.cuda.is_available() and device == "cuda":
        point_cloud = point_cloud.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and device == "mps":
        point_cloud = point_cloud.to('mps')
    
    # Generate uniform directions
    ambient_dimension = point_cloud.shape[1]
    directions = generate_uniform_directions(
        num_thetas=num_directions,
        d=ambient_dimension,
        device=point_cloud.device,
        seed=0
    )
    
    # Add batch dimension for processing
    batched_points = point_cloud.unsqueeze(0)  # Shape: [1, N, D]
    
    # Compute ECT for the point cloud
    # We need to reshape the input to match the expected format [BxN, D]
    batch_size = batched_points.shape[0]
    num_points = batched_points.shape[1]
    reshaped_points = batched_points.reshape(batch_size * num_points, ambient_dimension)
    
    # Create index tensor to identify which points belong to which batch
    index = torch.zeros(batch_size * num_points, dtype=torch.int32, device=point_cloud.device)
    
    # Compute ECT
    ect_result = compute_ect(
        reshaped_points,
        v=directions,
        radius=radius,
        resolution=resolution,
        scale=scale,
        index=index,
        ect_fn=indicator
    )
    
    return ect_result

def visualize_ect(ect_result, save_path=None):
    """
    Visualize the ECT result.
    
    Args:
        ect_result: torch.Tensor of shape [R, B, N] where R is resolution,
                   B is batch size, and N is number of directions
        save_path: Path to save the visualization
    """
    # Convert to numpy for visualization
    ect_np = ect_result.cpu().numpy()
    
    # Reshape for visualization
    resolution, batch_size, num_directions = ect_np.shape
    
    # Plot the ECT as a heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(ect_np[:, 0, :], aspect='auto', cmap='viridis')
    plt.colorbar(label='ECT Value')
    plt.xlabel('Direction')
    plt.ylabel('Resolution Step')
    plt.title('Euler Characteristic Transform')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def visualize_directions(directions, title="Learned Directions", save_path=None):
    """
    Visualize the direction vectors.
    
    Args:
        directions: torch.Tensor of directions
        title: Title for the plot
        save_path: Path to save the visualization
    """
    # Convert to numpy for visualization
    directions_np = directions.detach().cpu().numpy()
    
    # For 3D directions
    if directions_np.shape[0] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightgray', alpha=0.2)
        
        # Plot direction vectors
        ax.scatter(directions_np[0], directions_np[1], directions_np[2], c='red', s=50)
        
        # Connect to origin
        for i in range(directions_np.shape[1]):
            ax.plot([0, directions_np[0, i]], [0, directions_np[1, i]], [0, directions_np[2, i]], 'b-', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
    # For 2D directions
    elif directions_np.shape[0] == 2:
        plt.figure(figsize=(8, 8))
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        plt.plot(x, y, 'lightgray')
        
        # Plot direction vectors
        plt.scatter(directions_np[0], directions_np[1], c='red', s=50)
        
        # Connect to origin
        for i in range(directions_np.shape[1]):
            plt.plot([0, directions_np[0, i]], [0, directions_np[1, i]], 'b-', alpha=0.3)
        
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def learn_optimal_directions(point_cloud, num_directions=32, resolution=30, radius=1.0, scale=10.0, 
                           device="cpu", num_epochs=500, learning_rate=0.01):
    """
    Learn optimal directions for ECT computation on a point cloud.
    
    Args:
        point_cloud: torch.Tensor of shape [N, D]
        num_directions: Number of directions to learn
        resolution: Resolution for ECT computation
        radius: Radius for ECT computation
        scale: Scale factor for ECT computation
        device: Device to use for computation ('cpu', 'cuda', 'mps')
        num_epochs: Number of epochs for training
        learning_rate: Learning rate for optimizer
        
    Returns:
        tuple: (fixed_layer, learned_layer, loss_history)
    """
    # Move to specified device
    if torch.cuda.is_available() and device == "cuda":
        point_cloud = point_cloud.cuda()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and device == "mps":
        point_cloud = point_cloud.to('mps')
    
    # Convert to torch_geometric Data object
    point_cloud_data = Data(x=point_cloud)
    batch = Batch.from_data_list([point_cloud_data])
    
    # Get dimension
    ambient_dimension = point_cloud.shape[1]
    
    # Initialize the ground truth layer with uniform directions
    fixed_directions = generate_uniform_directions(
        num_thetas=num_directions,
        d=ambient_dimension,
        device=point_cloud.device,
        seed=0
    )
    
    fixed_layer = ECTLayer(
        ECTConfig(
            ect_type="points",
            resolution=resolution,
            scale=scale,
            radius=radius,
            normalized=False,
            fixed=True
        ),
        v=fixed_directions
    )
    
    # Generate initial directions for learnable layer (all in same direction)
    initial_directions = torch.zeros((ambient_dimension, num_directions), device=point_cloud.device)
    initial_directions[0, :] = 1.0  # All directions pointing along first axis
    
    # Initialize the learnable layer
    learnable_layer = ECTLayer(
        ECTConfig(
            ect_type="points",
            resolution=resolution,
            scale=scale,
            radius=radius,
            normalized=False,
            fixed=False
        ),
        v=initial_directions
    )
    
    # Compute ground truth ECT
    ect_groundtruth = fixed_layer(batch)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(learnable_layer.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    loss_history = []
    print("Starting direction learning...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        ect_pred = learnable_layer(batch)
        loss = loss_fn(ect_pred, ect_groundtruth)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")
    
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    return fixed_layer, learnable_layer, loss_history

def compare_ect_results(fixed_layer, learned_layer, batch, save_path=None):
    """
    Compare ECT results from fixed and learned directions.
    
    Args:
        fixed_layer: ECTLayer with fixed directions
        learned_layer: ECTLayer with learned directions
        batch: Batch of data
        save_path: Path to save the visualization
    """
    # Compute ECTs
    with torch.no_grad():
        ect_groundtruth = fixed_layer(batch)
        ect_learned = learned_layer(batch)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot ground truth ECT
    im0 = axes[0].imshow(ect_groundtruth.squeeze().cpu().numpy(), aspect='auto')
    axes[0].set_title('ECT with Uniform Directions')
    axes[0].set_xlabel('Direction')
    axes[0].set_ylabel('Resolution Step')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot learned ECT
    im1 = axes[1].imshow(ect_learned.squeeze().cpu().numpy(), aspect='auto')
    axes[1].set_title('ECT with Learned Directions')
    axes[1].set_xlabel('Direction')
    axes[1].set_ylabel('Resolution Step')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main():
    # Load mesh
    mesh_path = r'D:\Research\data\3d\human\thuman\all\THuman2.1_Release\model\0000\0000.obj'
    mesh = trimesh.load(mesh_path, process=False)
    
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    
    # Convert mesh to point cloud
    num_points = 2000
    point_cloud = mesh_to_point_cloud(mesh, num_points=num_points)
    print(f"Generated point cloud with {point_cloud.shape[0]} points")
    
    # Normalize point cloud
    normalized_point_cloud = normalize_point_cloud(point_cloud)
    print(f"Normalized point cloud: min={normalized_point_cloud.min().item():.4f}, max={normalized_point_cloud.max().item():.4f}")
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # First, let's compute ECT with fixed directions
    print("\n=== Computing ECT with fixed directions ===")
    ect_result = process_point_cloud_with_ect(
        normalized_point_cloud,
        num_directions=50,
        resolution=30,
        radius=1.0,
        scale=10.0,
        device=device
    )
    
    print(f"ECT result shape: {ect_result.shape}")
    
    # Visualize results
    save_path = os.path.join(os.path.dirname(__file__), "ect_fixed_directions.png")
    visualize_ect(ect_result, save_path)
    
    # Now, let's learn optimal directions
    print("\n=== Learning optimal directions for ECT ===")
    
    # Create Data object for torch_geometric
    point_cloud_data = Data(x=normalized_point_cloud.to(device))
    batch = Batch.from_data_list([point_cloud_data])
    
    # Learn directions
    fixed_layer, learned_layer, loss_history = learn_optimal_directions(
        normalized_point_cloud,
        num_directions=32,
        resolution=30,
        radius=1.0,
        scale=10.0,
        device=device,
        num_epochs=5000,
        learning_rate=0.01
    )
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Loss During Direction Learning')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "direction_learning_loss.png"))
    plt.show()
    
    # Visualize the learned directions
    visualize_directions(
        fixed_layer.v.movedim(-1, -2),
        title="Uniform Directions",
        save_path=os.path.join(os.path.dirname(__file__), "uniform_directions.png")
    )
    
    visualize_directions(
        learned_layer.v.movedim(-1, -2),
        title="Learned Directions",
        save_path=os.path.join(os.path.dirname(__file__), "learned_directions.png")
    )
    
    # Compare ECT results
    compare_ect_results(
        fixed_layer,
        learned_layer,
        batch,
        save_path=os.path.join(os.path.dirname(__file__), "ect_comparison.png")
    )
    
    # Optional: visualize the mesh
    if hasattr(mesh, 'visual') and mesh.visual.kind == 'texture':
        print("Texture found")
    
    visualize_mesh(mesh)

if __name__ == "__main__":
    main()