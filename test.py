import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

# ==========================================
# 1. Configuration & Reproducibility
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()
print(f"Using device: {DEVICE}")

# ==========================================
# 2. Data Preparation
# ==========================================

def get_dataset_stats(dataset_loader):
    """
    Computes mean and std of the dataset for normalization.
    """
    print("Computing dataset mean and std...")
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataset_loader:
        # data shape: [batch, 3, 32, 32]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print(f"Mean: {mean}, Std: {std}")
    return mean, std

def prepare_data():
    # 1. Initial transform to calculate mean/std (Resize + ToTensor)
    initial_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor() # Converts to [0,1]
    ])

    # Download Training Data
    train_set_raw = torchvision.datasets.STL10(
        root='./data', split='train', download=True, transform=initial_transform
    )
    
    # Create a temp loader to calc stats
    temp_loader = DataLoader(train_set_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    mean, std = get_dataset_stats(temp_loader)

    # 2. Final Transform with Normalization
    final_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Reload datasets with final transform
    train_dataset = torchvision.datasets.STL10(
        root='./data', split='train', download=True, transform=final_transform
    )
    test_dataset_full = torchvision.datasets.STL10(
        root='./data', split='test', download=True, transform=final_transform
    )

    # 3. Create Splits as per instructions
    # "Use: 300 test images for validation and 500 for testing"
    # We will take these from the provided test_dataset_full (which has 8000 images)
    
    # Generate indices
    total_test_imgs = len(test_dataset_full)
    indices = list(range(total_test_imgs))
    # Shuffle indices to get random selection
    random.shuffle(indices)
    
    val_indices = indices[:300]
    test_indices = indices[300:800] # Next 500 images
    
    val_dataset = Subset(test_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = train_dataset.classes
    return train_loader, val_loader, test_loader, classes

# ==========================================
# 3. Model Architecture (LeNet-5 Style)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # 1. Conv Layer 1: 6 filters, 5x5, stride 1
        # Input: 3x32x32 -> Output: 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        
        # Max Pool 1: 2x2, stride 2
        # Input: 6x28x28 -> Output: 6x14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Conv Layer 2: 16 filters, 5x5, stride 1
        # Input: 6x14x14 -> Output: 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Max Pool 2: 2x2, stride 2 (Same layer definition as above, used again)
        # Input: 16x10x10 -> Output: 16x5x5
        
        # Flatten for FC layers
        # 16 * 5 * 5 = 400
        
        # 3. FC Layer 1: 120 units
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        
        # 4. FC Layer 2: 84 units
        self.fc2 = nn.Linear(120, 84)
        
        # 5. Output Layer: 10 units
        self.fc3 = nn.Linear(84, 10)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Layer 1
        x = self.pool(self.relu(self.conv1(x)))
        # Layer 2
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # Layer 3
        x = self.relu(self.fc1(x))
        # Layer 4
        x = self.relu(self.fc2(x))
        # Output (No softmax, included in CrossEntropyLoss)
        x = self.fc3(x)
        return x

# ==========================================
# 4. Training & Validation Functions
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Log batch loss (optional: print every few batches to reduce clutter)
        # print(f"Batch {i}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ==========================================
# 5. Main Execution
# ==========================================

if __name__ == "__main__":
    # Load Data
    train_loader, val_loader, test_loader, classes = prepare_data()
    
    # Initialize Model
    model = LeNet5().to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Decay LR by 50% every 20 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Metrics Storage
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_epochs = []

    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        # Train
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        
        # Scheduler Step
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            v_loss, v_acc = validate(model, val_loader, criterion)
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            val_epochs.append(epoch + 1)
            print(f"--- Validation | Loss: {v_loss:.4f} | Acc: {v_acc:.2f}% ---")

    print("Training Complete.")

    # ==========================================
    # 6. Testing & Evaluation
    # ==========================================
    
    # Plotting Loss and Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Acc')
    plt.plot(val_epochs, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.show()

    # Final Evaluation on Test Set
    print("\nEvaluating on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Per-Class Accuracy
    # Diagonal elements / Sum of row
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"{classes[i]}: {acc * 100:.2f}%")