from torchvision import models
import torch.nn as nn
import torch
from PIL import Image
import torchvision.transforms as transforms

model = models.resnet50()
# Replace the final fully connected layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes for binary classification

device = torch.device('cpu')
model = model.to(device)

# Load the saved weights
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probabilities.cpu().numpy()

# Example usage
prediction, probs = predict_image(model, 'ss1.jpg', transforms)
print(f'Predicted class: {prediction}')
print(f'Probabilities: {probs}')