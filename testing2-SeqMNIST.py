import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from ncps.pytorch.AdaptiveLTC4 import LTCCell, AdaptiveLTCCell, AdaptiveLTC


# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Define the AdaptiveLTC network
input_size = 28  # Each row (or column) of the image as a sequence element
hidden_units = 100  # Example number of hidden units
num_classes = 10  # MNIST has 10 classes (digits 0-9)
adaptive_ltc_net = AdaptiveLTC(input_size, hidden_units, num_classes)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(adaptive_ltc_net.parameters(), lr=0.001)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images for sequence processing
        images = images.view(-1, 28, 28)  # Reshape to (batch_size, seq_len, input_size)

        # Forward pass
        outputs, _, _, _ = adaptive_ltc_net(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print/log loss
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# Test the model
adaptive_ltc_net.eval()  # Set the model to evaluation mode

with torch.no_grad():  # Inference mode, no need for gradients
    correct = 0
    total = 0
    for images, labels in test_loader:
        # Reshape images for sequence processing
        images = images.view(-1, 28, 28)  # Reshape to (batch_size, seq_len, input_size)

        # Forward pass
        outputs, _, _, _ = adaptive_ltc_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')


