import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import splitfolders
import kagglehub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, hamming_loss

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 配置参数
batch_size = 32
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 下载数据集
def download_dataset():
    print("开始下载数据集...")
    path = kagglehub.dataset_download("mohammedarfathr/orange-fruit-daatset")
    print(f"数据集下载路径: {path}")
    return path

# 数据集划分
def split_dataset(input_path, output_path):
    print(f"使用split-folders划分数据集...")
    # 如果输出目录不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 使用ratio方法按8:1:1比例划分数据集
    splitfolders.ratio(input_path, output=output_path, seed=1337, ratio=(0.8, 0.1, 0.1))
    print(f"数据集已划分到: {output_path}")

# 数据加载和转换
def load_data(data_path):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为224x224
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(            # 标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 创建数据集
    train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(data_path, 'val'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"类别: {train_dataset.classes}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, train_dataset.classes

# 构建分类模型
class OrangeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(OrangeClassifier, self).__init__()
        # 使用预训练的ResNet18作为基础模型
        self.model = resnet18(pretrained=True)
        # 更改最后的全连接层以匹配类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计损失和准确率
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')
    
    return model, history

# 评估模型
def evaluate_model(model, dataloader, device, classes):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # 计算各种评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'hamming_loss': hamming_loss(all_labels, all_preds)
    }
    
    print("\n模型评估指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
    print(f"Kappa系数 (Cohen's Kappa): {metrics['kappa']:.4f}")
    print(f"汉明损失 (Hamming Loss): {metrics['hamming_loss']:.4f}")
    
    return metrics

# 绘制训练历史图表
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # 下载数据集
    dataset_path = download_dataset()
    
    # 划分数据集
    split_path = "orange_split_dataset"
    split_dataset(dataset_path, split_path)
    
    # 加载数据
    train_loader, val_loader, test_loader, classes = load_data(split_path)
    
    # 创建模型
    model = OrangeClassifier(num_classes=len(classes))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # 评估模型
    evaluate_model(trained_model, test_loader, device, classes)
    
    # 绘制训练历史
    plot_history(history)
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'orange_classifier.pth')
    print("模型已保存为 'orange_classifier.pth'")

if __name__ == "__main__":
    main() 