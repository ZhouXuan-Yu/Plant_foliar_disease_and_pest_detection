import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import json
from pathlib import Path
import shutil

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orange_classifier.log')
    ]
)
logger = logging.getLogger("OrangeClassifier")

# 设置随机种子以确保结果可重复
def set_seed(seed=42):
    """设置随机种子，确保结果可重现"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")

# 配置参数类
class Config:
    def __init__(self):
        # 基本目录
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.output_dir = os.path.join(self.base_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练参数
        self.batch_size = 32
        self.num_epochs = 20
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 224
        self.dataset_name = "mohammedarfathr/orange-fruit-daatset"
        
        # 数据路径
        self.processed_dir = os.path.join(self.output_dir, "orange_processed_data")
        self.split_dir = os.path.join(self.output_dir, "orange_split_dataset")
        
        # 输出文件路径
        self.model_save_path = os.path.join(self.output_dir, "orange_classifier.pth")
        self.history_path = os.path.join(self.output_dir, "training_history.png")
        self.hires_history_path = os.path.join(self.output_dir, "training_history_hires.png")
        self.confusion_matrix_path = os.path.join(self.output_dir, "confusion_matrix.png")
        self.samples_path = os.path.join(self.output_dir, "training_samples.png")
        self.metrics_path = os.path.join(self.output_dir, "metrics.json")
        
        # 数据分割比例
        self.test_size = 0.2
        self.val_size = 0.1

    def __str__(self):
        return (f"配置信息:\n"
                f"- 批量大小: {self.batch_size}\n"
                f"- 训练轮数: {self.num_epochs}\n"
                f"- 学习率: {self.learning_rate}\n"
                f"- 设备: {self.device}\n"
                f"- 图像大小: {self.img_size}x{self.img_size}\n"
                f"- 数据集: {self.dataset_name}\n"
                f"- 输出目录: {self.output_dir}\n"
                f"- 分割目录: {self.split_dir}")

    def save_metrics(self, metrics, class_metrics=None, config_dict=None):
        """保存评估指标到JSON文件"""
        metrics_data = {k: float(v) for k, v in metrics.items()}
        
        if class_metrics:
            metrics_data['class_metrics'] = {}
            for cls_name, cls_metrics in class_metrics.items():
                metrics_data['class_metrics'][cls_name] = {
                    k: float(v) for k, v in cls_metrics.items()
                }
        
        if config_dict:
            metrics_data['config'] = config_dict
        
        with open(self.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型评估指标已保存到: {self.metrics_path}")

# 模型定义
class OrangeClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(OrangeClassifier, self).__init__()
        logger.info(f"创建OrangeClassifier模型, 类别数={num_classes}, 使用预训练={pretrained}")
        
        # 使用预训练的ResNet18作为基础模型
        self.model = resnet18(pretrained=pretrained)
        
        # 添加Dropout层以防止过拟合
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# 数据加载函数
def load_data(data_path, config):
    """加载已划分的数据集，并应用数据变换"""
    logger.info(f"加载数据集: {data_path}")
    
    try:
        # 定义训练数据转换
        train_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # 随机仿射变换
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 验证和测试数据转换（没有增强）
        eval_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 检查数据路径
        train_dir = os.path.join(data_path, 'train')
        val_dir = os.path.join(data_path, 'val')
        test_dir = os.path.join(data_path, 'test')
        
        logger.info(f"训练目录: {train_dir}")
        logger.info(f"验证目录: {val_dir}")
        logger.info(f"测试目录: {test_dir}")
        
        if not all(os.path.exists(p) for p in [train_dir, val_dir, test_dir]):
            missing = [p for p in [train_dir, val_dir, test_dir] if not os.path.exists(p)]
            logger.error(f"数据目录不存在: {missing}")
            return None, None, None, None
        
        # 检查每个目录中的类别
        for dir_name, dir_path in [("训练", train_dir), ("验证", val_dir), ("测试", test_dir)]:
            classes = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
            logger.info(f"{dir_name}目录中的类别: {classes}")
            
            if "FIELD IMAGES" in classes:
                logger.error(f"发现原始数据集结构，数据集可能未被正确处理。请检查 {dir_path}")
                return None, None, None, None
            
            for cls in classes:
                cls_path = os.path.join(dir_path, cls)
                img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
                logger.info(f"{dir_name}目录中类别 '{cls}' 包含 {len(img_files)} 张图像")
                
                if len(img_files) == 0:
                    logger.error(f"{dir_name}目录中类别 '{cls}' 没有图像文件")
                    return None, None, None, None
        
        # 创建数据集
        logger.info(f"创建训练数据集: {train_dir}")
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        
        logger.info(f"创建验证数据集: {val_dir}")
        val_dataset = ImageFolder(val_dir, transform=eval_transform)
        
        logger.info(f"创建测试数据集: {test_dir}")
        test_dataset = ImageFolder(test_dir, transform=eval_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if config.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if config.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True if config.device.type == 'cuda' else False
        )
        
        logger.info(f"类别: {train_dataset.classes}")
        logger.info(f"类别索引映射: {train_dataset.class_to_idx}")
        logger.info(f"训练集样本数: {len(train_dataset)}")
        logger.info(f"验证集样本数: {len(val_dataset)}")
        logger.info(f"测试集样本数: {len(test_dataset)}")
        
        # 检查是否有足够的样本
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
            logger.error("数据集中某些部分没有样本")
            return None, None, None, None
        
        # 检查是否有足够的类别
        if len(train_dataset.classes) < 2:
            logger.error(f"数据集类别数量不足: {len(train_dataset.classes)}")
            return None, None, None, None
        
        return train_loader, val_loader, test_loader, train_dataset.classes
    
    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None

# 可视化一个批次的图像
def visualize_batch(dataloader, classes, save_path):
    """可视化一个批次的图像"""
    try:
        # 获取一个批次的图像
        images, labels = next(iter(dataloader))
        
        # 定义反归一化变换
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # 创建图形
        plt.figure(figsize=(20, 10))
        
        # 显示每个图像
        for i in range(min(20, len(images))):
            # 反归一化
            img = inv_normalize(images[i])
            # 将张量转换为PIL图像
            img = img.permute(1, 2, 0).clamp(0, 1).numpy()
            
            # 显示图像
            plt.subplot(4, 5, i + 1)
            plt.imshow(img)
            plt.title(f"Class: {classes[labels[i]]}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(save_path)
        logger.info(f"样本批次可视化已保存到: {save_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"可视化数据批次时出错: {str(e)}")

# 绘制训练历史图表
def plot_history(history, save_path, hires_path):
    """绘制训练和验证的损失与准确率"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失', marker='o')
    plt.plot(history['val_loss'], label='验证损失', marker='s')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率', marker='o')
    plt.plot(history['val_acc'], label='验证准确率', marker='s')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"训练历史图表已保存到: {save_path}")
    
    # 将图表保存为高分辨率版本
    plt.savefig(hires_path, dpi=300)
    logger.info(f"高分辨率训练历史图表已保存到: {hires_path}")
    
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(model, dataloader, device, classes, save_path):
    """绘制混淆矩阵"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
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
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"混淆矩阵已保存到: {save_path}")
    plt.close() 