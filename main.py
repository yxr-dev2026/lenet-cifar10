import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

epoch_num = 10
for epoch in range(epoch_num):
    running_loss = 0.0
    net.train()
    for i, data in enumerate(train_loader):
        x, label = data
        x, label = x.to(device), label.to(device)
        opt.zero_grad()
        out = net(x)
        loss = criterion(out, label)
        loss.backward()
        opt.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.3f}")

    # 验证
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, label = data
            x, label = x.to(device), label.to(device)
            out = net(x)
            _, pred = torch.max(out,1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    print(f"Test Acc: {100*correct/total:.2f} %")

print("Finished")
