from torchvision.datasets import ImageFolder
from torchvision import models
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter

# model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)#models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
# print(model)

model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)#models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
print(model)



# Only freeze early layers, unfreeze later ones
# for name, param in model.named_parameters():
#     if 'layer4' in name or 'fc' in name:  # Unfreeze last block + classifier
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# Freeze feature extraction layers (optional - for faster training) # what is model.parameters() and what is requries_grad
for param in model.parameters():
    param.requires_grad = False

    # Unfreeze only the classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Replace the final fully connected layer for binary classification
num_features = model.classifier[-1].in_features
# model.classifier = nn.Linear(num_features, 2)  # 2 classes for binary classification
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),  # Keep this
    nn.Linear(num_features, 2)
)

# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize larger first
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Random crop with scaling
    # transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),  # People from overhead can be flipped vertically
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Random erasing
])

# No augmentation for validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder("./resNet_Images/train", transform=train_transforms)
val_dataset = ImageFolder("./resNet_Images/val", transform=val_transforms)
#test_dataset = ImageFolder("./resNet_Images/test")


# Add these debugging lines after creating your datasets
print("Dataset class mapping:")
print(f"Classes: {train_dataset.classes}")
print(f"Class to index: {train_dataset.class_to_idx}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Count samples per class
train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
val_labels = [val_dataset[i][1] for i in range(len(val_dataset))]
print(f"Training class distribution: {Counter(train_labels)}")
print(f"Validation class distribution: {Counter(val_labels)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cpu')
model = model.to(device)


# Based on how often each class appears in predictions
person_predictions = 211
not_person_predictions = 101

# Weight inversely to prediction frequency
class_weights = torch.tensor([1.0, 101/211])  # ≈ [1.0, 0.48]

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()#weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001, weight_decay=1e-4)  # Only train the classifier .. what is fc.parameters()

# Learning rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) #don't understand
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

def analyze_predictions(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    pred_counter = Counter(all_preds)
    label_counter = Counter(all_labels)
    
    print(f"Prediction distribution: {pred_counter}")
    print(f"True label distribution: {label_counter}")
    
    # Calculate per-class accuracy
    correct_0 = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    correct_1 = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    total_0 = label_counter[0]
    total_1 = label_counter[1]
    
    if total_0 > 0:
        print(f"Class 0 accuracy: {correct_0/total_0:.4f}")
    if total_1 > 0:
        print(f"Class 1 accuracy: {correct_1/total_1:.4f}")

def calculate_adaptive_weights(all_preds, smoothing_factor=1):
    """Calculate class weights based on prediction distribution"""
    pred_counter = Counter(all_preds)
    total_preds = len(all_preds)
    
    # Calculate prediction ratios
    pred_ratio_0 = pred_counter.get(0, 0) / total_preds
    pred_ratio_1 = pred_counter.get(1, 0) / total_preds
    
    # Inverse weighting with smoothing
    if pred_ratio_0 > 0 and pred_ratio_1 > 0:
        weight_0 = (0.5 / pred_ratio_0) * smoothing_factor + 1.0 * (1 - smoothing_factor)
        weight_1 = (0.5 / pred_ratio_1) * smoothing_factor + 1.0 * (1 - smoothing_factor)
    else:
        weight_0, weight_1 = 1.0, 1.0
    
    print("ASSIGNING WEIGHTS: " + str(weight_0) + " " + str(weight_1))
    
    return torch.tensor([weight_0, weight_1])

#training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        epoch_preds = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            epoch_preds.extend(preds.cpu().numpy())

        # Calculate new weights for next epoch
        if epoch < num_epochs - 1:  # Don't update on last epoch
            new_weights = calculate_adaptive_weights(epoch_preds)
            criterion = nn.CrossEntropyLoss(weight=new_weights)
            print(f"Updated class weights: {new_weights}")
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        analyze_predictions(model, val_loader, device)

        scheduler.step(val_epoch_loss)
        print()
    
    return model

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25)