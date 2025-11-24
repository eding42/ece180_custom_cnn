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
# 1. Configuration & Variations
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- VARIATION TOGGLES ---
# Set these to True to enable the specific variation for your experiments
USE_BATCH_NORM = False       # Variation 1
USE_L2_REGULARIZATION = False # Variation 2 (Weight Decay)
USE_DROPOUT = False          # Variation 3

# Configurable L2 strength (if enabled)
L2_LAMBDA = 1e-4 if USE_L2_REGULARIZATION else 0.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed()
print(f"Using device: {DEVICE}")
print(f"Experiment Config: Batch Norm={USE_BATCH_NORM}, L2={USE_L2_REGULARIZATION}, Dropout={USE_DROPOUT}")

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
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print(f"Mean: {mean}, Std: {std}")
    return mean, std

def prepare_data():
    initial_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    train_set_raw = torchvision.datasets.STL10(
        root='./data', split='train', download=True, transform=initial_transform
    )
    
    # Create a temp loader to calc stats
    temp_loader = DataLoader(train_set_raw, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    mean, std = get_dataset_stats(temp_loader)

    final_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = torchvision.datasets.STL10(
        root='./data', split='train', download=True, transform=final_transform
    )
    test_dataset_full = torchvision.datasets.STL10(
        root='./data', split='test', download=True, transform=final_transform
    )

    # Splits: 300 val, 500 test
    total_test_imgs = len(test_dataset_full)
    indices = list(range(total_test_imgs))
    random.shuffle(indices)
    
    val_indices = indices[:300]
    test_indices = indices[300:800]
    
    val_dataset = Subset(test_dataset_full, val_indices)
    test_dataset = Subset(test_dataset_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    classes = train_dataset.classes
    return train_loader, val_loader, test_loader, classes

# ==========================================
# 3. Model Architecture (LeNet-5 Style)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self, use_bn=False, use_dropout=False):
        super(LeNet5, self).__init__()
        self.use_bn = use_bn
        self.use_dropout = use_dropout

        # Layer 1: Conv 5x5
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(6)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv 5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(16)
        
        # FC Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.relu = nn.ReLU()
        
        # Dropout (Applied after FC layers typically)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.pool(self.relu(x))
        
        # Block 2
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.pool(self.relu(x))
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # FC 1
        x = self.fc1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        # FC 2
        x = self.fc2(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
            
        # Output
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
    train_loader, val_loader, test_loader, classes = prepare_data()
    
    # Initialize Model with toggles
    model = LeNet5(use_bn=USE_BATCH_NORM, use_dropout=USE_DROPOUT).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with optional L2 Regularization (weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_epochs = []

    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(t_loss)
        train_accs.append(t_acc)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if (epoch + 1) % 5 == 0:
            v_loss, v_acc = validate(model, val_loader, criterion)
            val_losses.append(v_loss)
            val_accs.append(v_acc)
            val_epochs.append(epoch + 1)
            print(f"--- Validation | Loss: {v_loss:.4f} | Acc: {v_acc:.2f}% ---")

    print("Training Complete.")

    # Plotting
    title_suffix = f"(BN={USE_BATCH_NORM}, L2={USE_L2_REGULARIZATION}, Drop={USE_DROPOUT})"
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss {title_suffix}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accs, label='Train Acc')
    plt.plot(val_epochs, val_accs, label='Val Acc', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy {title_suffix}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plots_variation.png')
    plt.show()

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

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.savefig('confusion_matrix.png')
    plt.show()

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"{classes[i]}: {acc * 100:.2f}%")