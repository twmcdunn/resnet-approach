import os
from collections import Counter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Check actual folder structure and counts
train_path = "./resNet_Images/train"
val_path = "./resNet_Images/val"

# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# No augmentation for validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_path, transform=train_transforms)
val_dataset = ImageFolder(val_path, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

print("=== FOLDER STRUCTURE ===")
for split, path in [("Train", train_path), ("Val", val_path)]:
    print(f"\n{split} folder contents:")
    if os.path.exists(path):
        folders = os.listdir(path)
        for folder in folders:
            folder_path = os.path.join(path, folder)
            if os.path.isdir(folder_path):
                count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {folder}: {count} images")
    else:
        print(f"  Path doesn't exist: {path}")

print("\n=== DATASET CLASS MAPPING ===")
print(f"Train classes: {train_dataset.classes}")
print(f"Train class_to_idx: {train_dataset.class_to_idx}")
print(f"Val classes: {val_dataset.classes}")
print(f"Val class_to_idx: {val_dataset.class_to_idx}")

print("\n=== ACTUAL TRAINING DATA DISTRIBUTION ===")
# Check what's actually being loaded
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
train_label_counts = Counter(train_labels)
print(f"Training label distribution: {train_label_counts}")

# Check a few sample images and their labels
print("\n=== SAMPLE TRAINING DATA ===")
for i in range(min(10, len(train_dataset))):
    img, label = train_dataset[i]
    print(f"Sample {i}: Label {label} (Class: {train_dataset.classes[label]})")

# Check training batch
print("\n=== FIRST TRAINING BATCH ===")
first_batch = next(iter(train_loader))
inputs, labels = first_batch
batch_label_counts = Counter(labels.numpy())
print(f"First batch label distribution: {batch_label_counts}")
print(f"First batch labels: {labels.numpy()}")

# Training loss analysis
print("\n=== TRAINING BEHAVIOR ANALYSIS ===")
print("If you see:")
print("- All training labels are 0: Your training data is imbalanced")
print("- Training accuracy is high but val accuracy is ~50%: Model is overfitting to majority class")
print("- Class 1 accuracy near 0%: Model never predicts class 1")