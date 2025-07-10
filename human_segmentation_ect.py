"""
Human Body Part Segmentation using Differentiable Euler Characteristic Transform (DECT)

This script implements body part segmentation for human point clouds using the
DECT framework. It leverages the topological properties captured by ECT to handle
variations in human body shapes, poses, and articulations.
"""

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
from dect.ect import compute_ect_points

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# HDF5 Dataset class
class HumanPointCloudDataset(Dataset):
    def __init__(self, hdf5_path, test_size=0.2, random_state=42, normalize=True):
        """
        Dataset for human point clouds with body part labels from HDF5 file
        
        Args:
            hdf5_path: Path to the HDF5 file
            test_size: Fraction of data to use for testing
            random_state: Random seed for train/test split
            normalize: Whether to normalize point coordinates
        """
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.samples = []
        
        # Load data from HDF5 file
        with h5py.File(hdf5_path, 'r') as f:
            # Get all the groups (each representing a human model)
            group_keys = list(f.keys())
            
            for group_key in group_keys:
                group = f[group_key]
                
                # Check if this group has both vertices and labels
                if 'sampled_vertices' in group and 'sampled_labels' in group:
                    vertices = np.array(group['sampled_vertices'][:])
                    labels = np.array(group['sampled_labels'][:])
                    
                    # Store data sample
                    self.samples.append({
                        'model_id': group_key,
                        'vertices': vertices,
                        'labels': labels
                    })
        
        # Train/test split at the model level
        train_indices, test_indices = train_test_split(
            range(len(self.samples)), 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Set the indices based on the split
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.is_train = True  # Default to training set
        
    def set_mode(self, train=True):
        """Switch between training and test sets"""
        self.is_train = train
    
    def __len__(self):
        """Return the number of samples in the current set"""
        return len(self.train_indices if self.is_train else self.test_indices)
    
    def __getitem__(self, idx):
        """Get a sample by index"""
        # Map the index to the corresponding sample
        sample_idx = self.train_indices[idx] if self.is_train else self.test_indices[idx]
        sample = self.samples[sample_idx]
        
        # Get vertices and labels
        vertices = sample['vertices'].astype(np.float32)
        labels = sample['labels'].astype(np.int64)
        
        # Normalize coordinates if specified
        if self.normalize:
            # Center and scale to [-1, 1]
            vertices_min = vertices.min(axis=0)
            vertices_max = vertices.max(axis=0)
            vertices = 2 * (vertices - vertices_min) / (vertices_max - vertices_min) - 1
        
        # Convert to PyTorch tensors
        vertices_tensor = torch.from_numpy(vertices)
        labels_tensor = torch.from_numpy(labels)
        
        # Create PyTorch Geometric Data object
        data = Data(x=vertices_tensor, y=labels_tensor)
        
        return data
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset"""
        all_labels = np.concatenate([sample['labels'] for sample in self.samples])
        return len(np.unique(all_labels))

    def visualize_sample(self, idx, train=True):
        """Visualize a sample from the dataset"""
        # Set the correct mode
        orig_mode = self.is_train
        self.set_mode(train)
        
        # Get the sample
        data = self[idx]
        vertices = data.x.numpy()
        labels = data.y.numpy()
        
        # Reset the mode
        self.set_mode(orig_mode)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the points, coloring by label
        scatter = ax.scatter(
            vertices[:, 0], 
            vertices[:, 1], 
            vertices[:, 2], 
            c=labels, 
            cmap='tab20', 
            s=5, 
            alpha=0.8
        )
        
        # Add colorbar and labels
        unique_labels = np.unique(labels)
        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_labels)
        cbar.set_label('Body Part Labels')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Human Point Cloud - {'Training' if train else 'Test'} Sample {idx}")
        
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

# Custom collate function for batching PyG Data objects
def collate_fn(batch):
    return Batch.from_data_list(batch)

# ECT-based Segmentation Model
class ECTSegmentationModel(nn.Module):
    def __init__(self, num_classes, num_thetas=64, resolution=32, radius=1.0, 
                 fixed_directions=False, device="cpu"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        
        # Initialize directions in 3D
        v = generate_uniform_directions(
            num_thetas=num_thetas, 
            d=3,  # 3D space
            seed=42, 
            device=device
        )
        
        # ECT layer (with learnable or fixed directions)
        self.ect_layer = ECTLayer(
            ECTConfig(
                ect_type="points",
                resolution=resolution,
                scale=8,
                radius=radius,
                normalized=True,
                fixed=fixed_directions
            ),
            v=v
        ).to(device)
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(device)
        
        # Calculate output size after feature extraction
        # For resolution=32, after 3 max pooling layers: 32 -> 16 -> 8 -> 4
        feature_output_size = resolution // (2**3)
        feature_dim = 128 * feature_output_size * feature_output_size
        
        # MLP for point-wise classification
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ).to(device)
        
    def forward(self, batch):
        """
        Forward pass for the ECT-based segmentation model
        
        Args:
            batch: Batch of PyG Data objects
            
        Returns:
            Tensor of shape (batch_size * num_points, num_classes) with segmentation logits
        """
        batch_size = batch.num_graphs
        
        # Compute ECT features
        ect_features = self.ect_layer(batch)  # Shape: (batch_size, 1, resolution, resolution)
        
        # Extract features using CNN
        features = self.feature_extractor(ect_features)
        
        # Flatten and pass through MLP
        features = features.view(batch_size, -1)
        logits = self.classifier(features)
        
        # Replicate for each point in the corresponding point cloud
        point_logits = []
        start_idx = 0
        
        for i, num_nodes in enumerate(batch.num_nodes.tolist()):
            # Replicate the logits for each point in this point cloud
            cloud_logits = logits[i].unsqueeze(0).expand(num_nodes, -1)
            point_logits.append(cloud_logits)
            start_idx += num_nodes
            
        # Concatenate all point logits
        all_point_logits = torch.cat(point_logits, dim=0)
        
        return all_point_logits

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=10, device="cpu"):
    """Train the ECT-based segmentation model"""
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            labels = batch.y
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track loss and accuracy
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                labels = batch.y
                
                outputs = model(batch)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"  Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # Save best model
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), 'best_ect_segmentation_model.pt')
            print("  Saved new best model")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, criterion, device="cpu"):
    """Evaluate the trained model on the test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y
            
            outputs = model(batch)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / total
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return test_loss, test_acc, all_preds, all_labels

def visualize_segmentation_results(model, dataset, indices, device="cpu"):
    """Visualize segmentation results on sample point clouds"""
    # Set dataset to test mode
    dataset.set_mode(False)
    model.eval()
    
    for idx in indices:
        # Get the sample
        data = dataset[idx].to(device)
        vertices = data.x.cpu().numpy()
        true_labels = data.y.cpu().numpy()
        
        # Get predictions
        with torch.no_grad():
            # Add batch dimension for the model
            batch = Batch.from_data_list([data])
            outputs = model(batch)
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.cpu().numpy()
        
        # Create 3D plots for both true labels and predictions
        fig = plt.figure(figsize=(15, 7))
        
        # True labels
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(
            vertices[:, 0], 
            vertices[:, 1], 
            vertices[:, 2], 
            c=true_labels, 
            cmap='tab20', 
            s=5, 
            alpha=0.8
        )
        ax1.set_title("Ground Truth Labels")
        plt.colorbar(scatter1, ax=ax1)
        
        # Predicted labels
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(
            vertices[:, 0], 
            vertices[:, 1], 
            vertices[:, 2], 
            c=predicted, 
            cmap='tab20', 
            s=5, 
            alpha=0.8
        )
        ax2.set_title("Predicted Labels")
        plt.colorbar(scatter2, ax=ax2)
        
        # Set equal aspect ratio for both plots
        for ax in [ax1, ax2]:
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
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig(f'segmentation_result_{idx}.png')
        plt.show()

def main():
    # Path to your HDF5 file
    hdf5_path = './data/human_pointclouds.hdf5'  # Update with your actual path
    
    # Hyperparameters
    batch_size = 4
    num_thetas = 64  # Number of directions for ECT
    resolution = 32  # Resolution of ECT grid
    radius = 1.0     # Radius for ECT calculation
    learning_rate = 0.001
    num_epochs = 20
    
    # Create dataset
    try:
        dataset = HumanPointCloudDataset(hdf5_path, normalize=True)
        num_classes = dataset.get_num_classes()
        print(f"Loaded dataset with {len(dataset.samples)} human models")
        print(f"Number of classes (body parts): {num_classes}")
        
        # Visualize a sample
        dataset.visualize_sample(0, train=True)
        
        # Create data loaders
        dataset.set_mode(True)  # Training mode
        train_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        dataset.set_mode(False)  # Test mode
        test_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # # Create model
        # model = ECTSegmentationModel(
        #     num_classes=num_classes,
        #     num_thetas=num_thetas,
        #     resolution=resolution,
        #     radius=radius,
        #     fixed_directions=False,  # Use learnable directions
        #     device=DEVICE
        # ).to(DEVICE)
        
        # # Define loss function and optimizer
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # # Train the model
        # print("\nTraining model...")
        # train_model(
        #     model, 
        #     train_loader, 
        #     test_loader,  # Using test set as validation for simplicity
        #     criterion, 
        #     optimizer, 
        #     num_epochs=num_epochs,
        #     device=DEVICE
        # )
        
        # # Evaluate the model
        # print("\nEvaluating model...")
        # test_loss, test_acc, all_preds, all_labels = evaluate_model(
        #     model, 
        #     test_loader, 
        #     criterion,
        #     device=DEVICE
        # )
        
        # # Visualize results on a few test samples
        # print("\nVisualizing segmentation results...")
        # visualize_segmentation_results(
        #     model, 
        #     dataset, 
        #     indices=[0, 1, 2],  # Visualize first three test samples
        #     device=DEVICE
        # )
        
    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {hdf5_path}")
        print("Please update the path to your HDF5 file.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
