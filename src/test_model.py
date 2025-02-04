import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import SimpleModel

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the latest trained model
model = SimpleModel()
model_path = sorted(glob("../models/*.pth"))[-1]  # Get the most recent model
model.load_state_dict(torch.load(model_path))
model.eval()

# Test accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Check if accuracy > 90% and parameters < 25,000
total_params = sum(p.numel() for p in model.parameters())
assert accuracy > 90, "Accuracy is less than 90%"
assert total_params < 25000, "Model has more than 25,000 parameters"
print("All tests passed!")