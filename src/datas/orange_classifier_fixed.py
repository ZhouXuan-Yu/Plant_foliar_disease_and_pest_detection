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
import splitfolders
import kagglehub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, hamming_loss
import logging
import time
import shutil
from pathlib import Path
from PIL import Image
import matplotlib as mpl

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"随机种子已设置为: {seed}")

# 配置参数
class Config:
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 1
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = 224
        self.dataset_name = "mohammedarfathr/orange-fruit-daatset"
        self.processed_dir = "orange_processed_data"
        self.split_dir = "orange_split_dataset"
        self.model_save_path = "orange_classifier.pth"
        self.plot_save_path = "training_history.png"
        self.test_size = 0.1
        self.val_size = 0.1

    def __str__(self):
        return f"配置信息:\n" + \
               f"- 批量大小: {self.batch_size}\n" + \
               f"- 训练轮数: {self.num_epochs}\n" + \
               f"- 学习率: {self.learning_rate}\n" + \
               f"- 设备: {self.device}\n" + \
               f"- 图像大小: {self.img_size}x{self.img_size}\n" + \
               f"- 数据集: {self.dataset_name}\n" + \
               f"- 分割目录: {self.split_dir}"

# 下载数据集
def download_dataset(config):
    """下载橙子数据集"""
    logger.info(f"开始下载数据集: {config.dataset_name}")
    
    try:
        path = kagglehub.dataset_download(config.dataset_name)
        logger.info(f"数据集下载路径: {path}")
        
        # 列出下载的文件和目录
        try:
            all_items = os.listdir(path)
            logger.info(f"下载的数据集包含以下内容: {all_items}")
            
            # 检查是否有目录
            dirs = [item for item in all_items if os.path.isdir(os.path.join(path, item))]
            logger.info(f"下载的数据集中的目录: {dirs}")
            
            # 检查下载的数据集结构
            field_images_dir = os.path.join(path, "FIELD IMAGES")
            if os.path.exists(field_images_dir):
                logger.info(f"找到'FIELD IMAGES'目录: {field_images_dir}")
                
                # 列出包含的类别
                classes = [d for d in os.listdir(field_images_dir) 
                          if os.path.isdir(os.path.join(field_images_dir, d))]
                logger.info(f"'FIELD IMAGES'目录中的类别: {classes}")
                
                # 检查每个类别的图像数量
                total_images = 0
                for cls in classes:
                    cls_dir = os.path.join(field_images_dir, cls)
                    img_files = [f for f in os.listdir(cls_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
                    logger.info(f"类别 '{cls}' 包含 {len(img_files)} 张图像")
                    total_images += len(img_files)
                
                logger.info(f"数据集总共包含 {total_images} 张图像")
                
                if total_images == 0:
                    logger.error("数据集中没有找到任何图像")
                    return None
            else:
                # 如果没有'FIELD IMAGES'目录，检查是否有直接的类别目录
                potential_classes = [d for d in dirs if not d.startswith('.')]
                
                if not potential_classes:
                    logger.error(f"找不到'FIELD IMAGES'目录，也没有找到其他可能的类别目录")
                    logger.info(f"下载的目录结构: {all_items}")
                    return None
                
                logger.info(f"可能的类别目录: {potential_classes}")
                
                # 检查这些目录是否包含图像
                total_images = 0
                for cls in potential_classes:
                    cls_dir = os.path.join(path, cls)
                    if os.path.isdir(cls_dir):
                        img_files = [f for f in os.listdir(cls_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
                        logger.info(f"目录 '{cls}' 包含 {len(img_files)} 张图像")
                        total_images += len(img_files)
                
                logger.info(f"数据集总共包含 {total_images} 张图像")
                
                if total_images == 0:
                    logger.error("数据集中没有找到任何图像")
                    return None
        except Exception as e:
            logger.error(f"检查下载的数据集时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return path
    except Exception as e:
        logger.error(f"下载数据集时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# 处理数据集，使其适合ImageFolder的格式
def process_dataset(download_path, output_path, config):
    """将下载的数据集转换为简单的类别/图像结构"""
    logger.info(f"开始处理数据集...")
    logger.info(f"下载路径: {download_path}")
    logger.info(f"输出路径: {output_path}")
    
    # 检查下载路径是否存在
    if not os.path.exists(download_path):
        logger.error(f"下载路径不存在: {download_path}")
        return False
    
    # 确保输出目录存在
    if os.path.exists(output_path):
        logger.warning(f"输出目录已存在，将被删除: {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    # 源目录
    source_dir = os.path.join(download_path, "FIELD IMAGES")
    if not os.path.exists(source_dir):
        source_dir = download_path  # 尝试直接使用下载路径
        logger.warning(f"'FIELD IMAGES'目录不存在，尝试直接使用下载路径: {source_dir}")
        
        # 检查下载路径是否包含类别目录
        subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        logger.info(f"在下载路径中找到以下子目录: {subdirs}")
        
        if "FIELD IMAGES" in subdirs:
            source_dir = os.path.join(download_path, "FIELD IMAGES")
            logger.info(f"找到'FIELD IMAGES'目录，使用: {source_dir}")
    
    if not os.path.exists(source_dir):
        logger.error(f"源目录不存在: {source_dir}")
        return False
    
    # 列出所有类别
    classes = [d for d in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, d))]
    
    logger.info(f"在源目录中找到以下类别: {classes}")
    
    if not classes:
        logger.error(f"在 {source_dir} 中找不到任何类别目录")
        return False
    
    # 复制每个类别的图像
    for cls in classes:
        src_cls_dir = os.path.join(source_dir, cls)
        dst_cls_dir = os.path.join(output_path, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        logger.info(f"处理类别 '{cls}'...")
        
        # 查找所有图像文件
        img_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif']:
            img_files.extend(list(Path(src_cls_dir).glob(f'*{ext}')))
            img_files.extend(list(Path(src_cls_dir).glob(f'*{ext.upper()}')))
        
        logger.info(f"类别 '{cls}' 中找到 {len(img_files)} 张图像")
        
        if not img_files:
            logger.warning(f"类别 '{cls}' 中找不到任何图像文件")
            continue
        
        # 复制图像文件
        valid_images = 0
        for img_path in img_files:
            dst_path = os.path.join(dst_cls_dir, img_path.name)
            try:
                # 检查图像是否有效
                with Image.open(img_path) as img:
                    img.verify()  # 验证图像
                shutil.copy2(img_path, dst_path)
                valid_images += 1
            except Exception as e:
                logger.warning(f"复制图像 {img_path} 时出错: {str(e)}")
        
        # 检查是否复制了任何图像
        copied_files = os.listdir(dst_cls_dir)
        logger.info(f"类别 '{cls}': 成功复制了 {len(copied_files)} 张有效图像 (总共尝试 {len(img_files)} 张)")
        
        if not copied_files:
            logger.error(f"类别 '{cls}' 没有复制任何图像文件")
            return False
    
    # 最终检查
    all_classes = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    logger.info(f"成功处理类别: {all_classes}")
    
    if len(all_classes) < 2:
        logger.error(f"处理后的类别数量不足: {len(all_classes)}")
        return False
    
    logger.info(f"数据集处理完成，保存到: {output_path}")
    return True

# 数据集划分
def split_dataset(input_path, output_path, config):
    """使用split-folders划分数据集为训练/验证/测试集"""
    logger.info(f"使用split-folders划分数据集...")
    logger.info(f"输入目录: {input_path}")
    logger.info(f"输出目录: {output_path}")
    
    # 检查输入目录
    if not os.path.exists(input_path):
        logger.error(f"输入目录不存在: {input_path}")
        return False
    
    # 检查输入目录中是否有足够的类别和图像
    classes = [d for d in os.listdir(input_path) 
              if os.path.isdir(os.path.join(input_path, d))]
    
    logger.info(f"在输入目录中找到以下类别: {classes}")
    
    if not classes:
        logger.error(f"在 {input_path} 中找不到任何类别目录")
        return False
    
    # 检查是否有原始的'FIELD IMAGES'目录，这可能导致问题
    if "FIELD IMAGES" in classes:
        logger.error(f"在输入目录中发现原始的'FIELD IMAGES'目录，这表明数据集未被正确处理")
        return False
    
    # 检查每个类别是否有足够的图像
    valid_classes = True
    for cls in classes:
        cls_dir = os.path.join(input_path, cls)
        img_files = [f for f in os.listdir(cls_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
        logger.info(f"类别 '{cls}' 包含 {len(img_files)} 张图像")
        
        if len(img_files) == 0:
            logger.error(f"类别 '{cls}' 中没有图像文件")
            valid_classes = False
    
    if not valid_classes:
        logger.error("一些类别目录中没有找到图像文件")
        return False
    
    # 删除已存在的输出目录
    if os.path.exists(output_path):
        logger.warning(f"输出目录已存在，将被删除: {output_path}")
        shutil.rmtree(output_path)
    
    try:
        # 使用ratio方法按比例划分数据集
        logger.info(f"开始划分数据集，比例为: 训练={1 - config.test_size - config.val_size}, 验证={config.val_size}, 测试={config.test_size}")
        splitfolders.ratio(
            input_path, 
            output=output_path, 
            seed=42, 
            ratio=(1 - config.test_size - config.val_size, config.val_size, config.test_size)
        )
        
        # 检查划分后的数据集
        datasets = ['train', 'val', 'test']
        for dataset in datasets:
            dataset_dir = os.path.join(output_path, dataset)
            if not os.path.exists(dataset_dir):
                logger.error(f"{dataset} 目录不存在: {dataset_dir}")
                return False
            
            dataset_classes = [d for d in os.listdir(dataset_dir) 
                              if os.path.isdir(os.path.join(dataset_dir, d))]
            
            if not dataset_classes:
                logger.error(f"在 {dataset_dir} 中找不到任何类别目录")
                return False
            
            for cls in dataset_classes:
                cls_dir = os.path.join(dataset_dir, cls)
                img_files = [f for f in os.listdir(cls_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
                logger.info(f"{dataset} 集中类别 '{cls}' 包含 {len(img_files)} 张图像")
                
                if len(img_files) == 0:
                    logger.error(f"{dataset} 集中类别 '{cls}' 没有图像文件")
                    return False
        
        logger.info(f"数据集已成功划分到: {output_path}")
        return True
    except Exception as e:
        logger.error(f"划分数据集时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# 数据加载和转换
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

# 构建分类模型
class OrangeClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
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

# 可视化一个批次的图像
def visualize_batch(dataloader, classes, save_path="sample_batch.png"):
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

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path="best_model.pth"):
    """训练模型并保存最佳模型"""
    logger.info(f"开始训练模型, epochs={num_epochs}")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info('-' * 30)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start_time = time.time()
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
            
            batch_time = time.time() - batch_start_time
            
            # 每10个批次打印一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, "
                           f"Acc: {100.0 * correct/total:.2f}%, "
                           f"Time: {batch_time:.2f}s")
        
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
        
        # 更新学习率
        if scheduler:
            scheduler.step(val_epoch_loss)
            logger.info(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"  训练 - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        logger.info(f"  验证 - Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}")
        logger.info(f"  Epoch耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            logger.info(f"  验证准确率提升: {best_val_acc:.4f} -> {val_epoch_acc:.4f}")
            best_val_acc = val_epoch_acc
            best_model_wts = model.state_dict().copy()
            torch.save(best_model_wts, save_path)
            logger.info(f"  保存最佳模型到: {save_path}")
    
    # 训练完成，加载最佳模型
    if best_model_wts:
        model.load_state_dict(best_model_wts)
        logger.info(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
    
    return model, history

# 评估模型
def evaluate_model(model, dataloader, device, classes):
    """详细评估模型性能"""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    logger.info("开始评估模型...")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 将列表转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 计算各种评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'hamming_loss': hamming_loss(all_labels, all_preds)
    }
    
    logger.info("\n模型评估指标:")
    logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
    logger.info(f"召回率 (Recall): {metrics['recall']:.4f}")
    logger.info(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
    logger.info(f"Kappa系数 (Cohen's Kappa): {metrics['kappa']:.4f}")
    logger.info(f"汉明损失 (Hamming Loss): {metrics['hamming_loss']:.4f}")
    
    # 生成每个类别的性能指标
    class_metrics = {}
    for cls_idx, cls_name in enumerate(classes):
        # 创建二进制标签和预测
        cls_labels = (all_labels == cls_idx).astype(int)
        cls_preds = (all_preds == cls_idx).astype(int)
        
        # 如果类别中有样本，则计算相关指标
        if np.sum(cls_labels) > 0:
            class_metrics[cls_name] = {
                'precision': precision_score(cls_labels, cls_preds, zero_division=0),
                'recall': recall_score(cls_labels, cls_preds, zero_division=0),
                'f1': f1_score(cls_labels, cls_preds, zero_division=0),
                'support': np.sum(cls_labels),
                'correct': np.sum(cls_labels & cls_preds)
            }
            
            logger.info(f"\n类别 '{cls_name}' 评估指标:")
            logger.info(f"  样本数: {class_metrics[cls_name]['support']}")
            logger.info(f"  正确预测: {class_metrics[cls_name]['correct']}")
            logger.info(f"  准确率: {class_metrics[cls_name]['correct'] / class_metrics[cls_name]['support']:.4f}")
            logger.info(f"  精确率: {class_metrics[cls_name]['precision']:.4f}")
            logger.info(f"  召回率: {class_metrics[cls_name]['recall']:.4f}")
            logger.info(f"  F1分数: {class_metrics[cls_name]['f1']:.4f}")
    
    return metrics, class_metrics

# 绘制训练历史图表
def plot_history(history, save_path="training_history.png"):
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
    plt.savefig(save_path.replace('.png', '_hires.png'), dpi=300)
    
    plt.close()

# 绘制混淆矩阵
def plot_confusion_matrix(model, dataloader, device, classes, save_path="confusion_matrix.png"):
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

def train():
    """训练模型的主函数"""
    try:
        # 设置随机种子
        set_seed(42)
        
        # 配置
        config = Config()
        logger.info(str(config))
        
        # 检查各个步骤的数据目录是否存在
        split_dir_valid = False
        if os.path.exists(config.split_dir):
            # 检查是否有 train, val, test 子目录
            has_all_subsets = all(os.path.exists(os.path.join(config.split_dir, d)) for d in ['train', 'val', 'test'])
            if has_all_subsets:
                # 检查每个子目录中是否有类别子目录
                all_subset_valid = True
                total_classes = 0
                for subset in ['train', 'val', 'test']:
                    subset_dir = os.path.join(config.split_dir, subset)
                    classes = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
                    if not classes or len(classes) < 2:
                        logger.warning(f"{subset} 目录中没有足够的类别: {classes}")
                        all_subset_valid = False
                        break
                    elif subset == 'train':  # 只检查训练集中的类别数
                        total_classes = len(classes)
                
                if all_subset_valid and total_classes >= 2:
                    split_dir_valid = True
                    logger.info(f"发现有效的已划分数据集目录: {config.split_dir}")
            else:
                logger.warning(f"数据集目录 {config.split_dir} 中缺少训练、验证或测试子目录")
        
        if not split_dir_valid:
            logger.error("数据集目录无效，请先运行数据处理步骤")
            return
        
        # 加载数据
        logger.info(f"开始加载已准备好的数据集: {config.split_dir}")
        data_loaders = load_data(config.split_dir, config)
        if None in data_loaders:
            logger.error("数据加载失败，程序终止")
            return
        
        train_loader, val_loader, test_loader, classes = data_loaders
        
        # 可视化部分数据
        visualize_batch(train_loader, classes, "training_samples.png")
        
        # 创建模型
        num_classes = len(classes)
        logger.info(f"创建模型: 类别数={num_classes}")
        model = OrangeClassifier(num_classes=num_classes, pretrained=True)
        model = model.to(config.device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        
        # 学习率调度器 (ReduceLROnPlateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3, verbose=True
        )
        
        # 训练模型
        logger.info(f"开始训练模型，模型将保存到: {config.model_save_path}")
        trained_model, history = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            config.num_epochs, config.device, config.model_save_path
        )
        
        # 绘制训练历史
        plot_history(history, config.plot_save_path)
        
        # 在测试集上评估模型
        metrics, class_metrics = evaluate_model(trained_model, test_loader, config.device, classes)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(trained_model, test_loader, config.device, classes, "confusion_matrix.png")
        
        logger.info("训练完成!")
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def test():
    """测试模型的主函数"""
    try:
        config = Config()
        
        # 检查模型文件是否存在
        if not os.path.exists(config.model_save_path):
            logger.error(f"找不到模型文件: {config.model_save_path}")
            logger.error("请先运行训练步骤")
            return
        
        # 加载数据
        logger.info(f"开始加载测试数据集: {config.split_dir}")
        data_loaders = load_data(config.split_dir, config)
        if None in data_loaders:
            logger.error("数据加载失败，程序终止")
            return
        
        _, _, test_loader, classes = data_loaders
        
        # 加载模型
        logger.info(f"加载模型: {config.model_save_path}")
        model = OrangeClassifier(num_classes=len(classes), pretrained=False)
        model.load_state_dict(torch.load(config.model_save_path))
        model = model.to(config.device)
        model.eval()
        
        # 评估模型
        metrics, class_metrics = evaluate_model(model, test_loader, config.device, classes)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(model, test_loader, config.device, classes, "confusion_matrix.png")
        
        logger.info("测试完成!")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        train() 