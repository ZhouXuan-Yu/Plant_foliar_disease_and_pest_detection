import os
import sys
import torch
import numpy as np
import argparse
import time
import logging
import traceback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, hamming_loss

# 导入公共模块
from orange_classifier_common import (
    set_seed, Config, OrangeClassifier, 
    load_data, plot_confusion_matrix, logger
)

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
            # 计算准确率
            accuracy = np.sum(cls_labels & cls_preds) / np.sum(cls_labels)
            
            # 计算类别指标
            class_metrics[cls_name] = {
                'precision': precision_score(cls_labels, cls_preds, zero_division=0),
                'recall': recall_score(cls_labels, cls_preds, zero_division=0),
                'f1': f1_score(cls_labels, cls_preds, zero_division=0),
                'support': int(np.sum(cls_labels)),
                'correct': int(np.sum(cls_labels & cls_preds)),
                'accuracy': accuracy
            }
            
            logger.info(f"\n类别 '{cls_name}' 评估指标:")
            logger.info(f"  样本数: {class_metrics[cls_name]['support']}")
            logger.info(f"  正确预测: {class_metrics[cls_name]['correct']}")
            logger.info(f"  准确率: {accuracy:.4f}")
            logger.info(f"  精确率: {class_metrics[cls_name]['precision']:.4f}")
            logger.info(f"  召回率: {class_metrics[cls_name]['recall']:.4f}")
            logger.info(f"  F1分数: {class_metrics[cls_name]['f1']:.4f}")
    
    return metrics, class_metrics

def test():
    """测试模型的主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='橙子叶片分类模型评估')
        parser.add_argument('--model', type=str, help='模型路径')
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
        logger.info(str(config))
        
        # 如果指定了模型路径，则使用指定的路径
        if args.model:
            model_path = args.model
            logger.info(f"使用指定的模型路径: {model_path}")
        else:
            model_path = config.model_save_path
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"找不到模型文件: {model_path}")
            logger.error("请先运行 orange_classifier_train.py 进行模型训练")
            return
        
        # 检查已划分的数据目录是否存在
        if not os.path.exists(config.split_dir):
            logger.error(f"找不到数据集目录: {config.split_dir}")
            logger.error("请先运行 orange_classifier_train.py 准备数据")
            return
        
        # 加载数据
        logger.info(f"加载测试数据集: {config.split_dir}")
        data_loaders = load_data(config.split_dir, config)
        if None in data_loaders:
            logger.error("数据加载失败，程序终止")
            return
        
        train_loader, val_loader, test_loader, classes = data_loaders
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        num_classes = len(classes)
        model = OrangeClassifier(num_classes=num_classes, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model = model.to(config.device)
        model.eval()
        
        # 评估模型
        metrics, class_metrics = evaluate_model(model, test_loader, config.device, classes)
        
        # 绘制混淆矩阵
        logger.info("绘制混淆矩阵...")
        plot_confusion_matrix(model, test_loader, config.device, classes, config.confusion_matrix_path)
        
        # 保存配置和指标
        config_dict = {
            "num_classes": num_classes,
            "device": str(config.device),
            "img_size": config.img_size,
            "classes": classes,
            "model_path": model_path,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存评估指标
        config.save_metrics(metrics, class_metrics, config_dict)
        
        logger.info("评估完成! 所有指标和图表已保存")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test() 