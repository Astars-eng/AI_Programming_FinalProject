import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 配置设备：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

batch_size = 4
transform = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载CIFAR10数据集（指定到数据盘，避免系统盘满）
trainset = torchvision.datasets.CIFAR10(
    root='/root/autodl-tmp/Creating_Illusion/data',  # 改为你的数据盘路径
    train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/root/autodl-tmp/Creating_Illusion/data',  # 改为你的数据盘路径
    train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn

class LeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

# 模型移到GPU
model = LeNet().to(device)
# 降低学习率，添加动量（提升收敛性）
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)  
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter('/root/autodl-tmp/Creating_Illusion/runs/cifar_lenet_experiment')  # 数据盘路径

if __name__ == '__main__':
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()  # 显式设置训练模式
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 数据移到GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 每1000个批次记录一次损失
            if i % 1000 == 999:
                avg_loss = running_loss / 1000
                print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss:.3f}')
                
                global_step = epoch * len(trainloader) + i
                writer.add_scalar('training loss', avg_loss, global_step)
                
                running_loss = 0.0

    print('Finished Training!')
    writer.close()

    # 测试阶段
    model.eval()  # 显式设置评估模式
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 修复批次边界问题：只遍历有效标签
            for label, pred in zip(labels, predicted):
                class_correct[label] += (pred == label).item()
                class_total[label] += 1

    print(f"Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%")
    for i in range(10):
        if class_total[i] > 0:  # 避免除以0
            print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
        else:
            print(f'Accuracy of {classes[i]}: 0.00%')