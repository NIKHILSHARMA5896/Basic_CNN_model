import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import os  # Import os module to handle directory creation

# Define a simple model with <25,000 parameters
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 30)  # Reduced to 30 neurons
        self.fc2 = nn.Linear(30, 16)       # Reduced to 16 neurons
        self.fc3 = nn.Linear(16, 10)       # Output layer remains the same

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss, and optimizer
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train for 1 epoch
for epoch in range(1):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Create the models directory if it doesn't exist
os.makedirs("../models", exist_ok=True)

# Save the model with a timestamp suffix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"../models/mnist_model_{timestamp}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Print total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")