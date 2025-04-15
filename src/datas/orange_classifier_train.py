import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import traceback
import logging
import argparse
import kagglehub
import splitfolders
import shutil
from pathlib import Path
from PIL import Image

# 导入公共模块
from orange_classifier_common import (
    set_seed, Config, OrangeClassifier, 
    load_data, visualize_batch, plot_history,
    plot_confusion_matrix, logger
)

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
            logger.error(traceback.format_exc())
        
        return path
    except Exception as e:
        logger.error(f"下载数据集时出错: {str(e)}")
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
        logger.error(traceback.format_exc())
        return False

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
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

def train():
    """训练模型的主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='橙子叶片分类模型训练')
        parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
        parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
        parser.add_argument('--lr', type=float, default=0.001, help='学习率')
        parser.add_argument('--download', action='store_true', help='重新下载数据集')
        parser.add_argument('--process', action='store_true', help='重新处理数据集')
        parser.add_argument('--split', action='store_true', help='重新划分数据集')
        parser.add_argument('--debug', action='store_true', help='开启调试模式')
        args = parser.parse_args()
        
        # 设置日志级别
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.info("调试模式已启用")
        
        # 设置随机种子
        set_seed(42)
        
        # 配置
        config = Config()
        config.num_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        logger.info(str(config))
        
        # 下载和处理数据集
        download_path = None
        processed_dir_valid = False
        
        # 检查处理后的数据目录是否存在
        if os.path.exists(config.processed_dir):
            # 检查是否有类别子目录
            classes = [d for d in os.listdir(config.processed_dir) 
                      if os.path.isdir(os.path.join(config.processed_dir, d))]
            if classes and len(classes) >= 2:
                has_images = True
                for cls in classes:
                    cls_dir = os.path.join(config.processed_dir, cls)
                    images = [f for f in os.listdir(cls_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'))]
                    if not images:
                        has_images = False
                        break
                
                if has_images:
                    processed_dir_valid = True
                    logger.info(f"发现有效的处理后数据目录: {config.processed_dir}")
        
        if args.download or not processed_dir_valid:
            logger.info("正在下载数据集...")
            download_path = download_dataset(config)
            if not download_path:
                logger.error("下载数据集失败，程序终止")
                return
            
            logger.info("正在处理数据集...")
            if not process_dataset(download_path, config.processed_dir, config):
                logger.error("处理数据集失败，程序终止")
                return
        
        # 检查已划分的数据目录是否存在
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
        
        if args.split or not split_dir_valid:
            logger.info("正在划分数据集...")
            if not split_dataset(config.processed_dir, config.split_dir, config):
                logger.error("划分数据集失败，程序终止")
                return
        
        # 加载数据
        logger.info(f"开始加载已准备好的数据集: {config.split_dir}")
        data_loaders = load_data(config.split_dir, config)
        if None in data_loaders:
            logger.error("数据加载失败，程序终止")
            return
        
        train_loader, val_loader, test_loader, classes = data_loaders
        
        # 可视化部分数据
        visualize_batch(train_loader, classes, config.samples_path)
        
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
        plot_history(history, config.history_path, config.hires_history_path)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(trained_model, val_loader, config.device, classes, config.confusion_matrix_path)
        
        logger.info("训练完成! 请运行 orange_classifier_test.py 进行模型评估")
        
        # 保存训练配置
        config_dict = {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "device": str(config.device),
            "img_size": config.img_size,
            "classes": classes,
            "dataset_name": config.dataset_name,
            "image_size": f"{config.img_size}x{config.img_size}",
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 创建基本指标文件
        basic_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "kappa": 0.0,
            "hamming_loss": 0.0,
            "config": config_dict
        }
        config.save_metrics(basic_metrics)
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    train() 