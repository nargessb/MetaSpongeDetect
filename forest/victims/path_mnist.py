# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from medmnist import INFO
from medmnist.dataset import PathMNIST

# ----------------- Config --------------------
data_flag = 'pathmnist'
download = True
batch_size = 64
num_epochs = 5
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dataset Info ----------------
info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = PathMNIST(split='train', transform=transform, download=download)
test_dataset = PathMNIST(split='test', transform=transform, download=download)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------- Model + Hook Logic -------------
model = resnet18(num_classes=n_classes)
model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.to(device)

# Register hook to count neuron firing
def hook_fn(module, input, output):
    activation = output.detach().cpu()
    print(f"Activation stats - mean: {activation.mean():.4f}, nonzero ratio: {(activation != 0).float().mean():.4f}")

hook = model.layer1[0].register_forward_hook(hook_fn)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ----------------- Training -------------------
print("Training started...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.squeeze().long().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# ----------------- Evaluation ------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

# ----------------- Visualization ------------------
sample_img, sample_label = train_dataset[0]
plt.imshow(sample_img.permute(1, 2, 0))
plt.title(f"Label: {sample_label.item()}")
plt.axis('off')
plt.show()

# Clean up
hook.remove()