# 植物叶片病害与害虫检测系统

## 📝 项目概述

基于神经网络的植物叶片病害与害虫智能检测系统，帮助农户实时监测作物健康状况，提早发现病虫害，提供科学防治建议，助力现代农业生产。

## ✨ 主要功能

- **多输入支持**：支持图像、视频、摄像头实时检测
- **智能识别**：高精度识别多种植物病害与害虫，精准定位受损区域
- **图像分割**：精确分离健康区域与病害区域，评估受损程度
- **自动分析报告**：自动生成检测结果报告，提供防治建议
- **灵活部署**：支持本地部署、服务器部署，适应不同应用场景
- **用户友好界面**：提供Web界面和桌面应用，操作简单直观

streamlit run orange_classifier_web.py

# 类别

names:
  0: 病害类别1
  1: 病害类别2
  2: 害虫类别1

# ...添加更多类别

```
3. 开始训练
```bash
python orange_classifier_train.py --epochs 100 --debug  
```

4. 模型测试
   
   ```bash
   python orange_classifier_test.py 
   ```

#### 分类模型

1. 组织数据集，每个类别一个文件夹
   
   ```
   dataset/
   ├── 类别1/
   │   ├── image1.jpg
   │   └── ...
   ├── 类别2/
   │   ├── image1.jpg
   │   └── ...
   └── ...
   ```

## 📋 支持的病害类别

系统目前支持以下常见植物病害和害虫的检测：

## 🔧 技术架构

- **前端模块**：Streamlit Web界面 
- **模型模块**：基于神经网络分类训练
- **后端模块**：基于Python的图像处理与结果分析系统
- **训练模块**：支持自定义数据集训练，迁移学习

## 启动可视化页面

src\datas 文件夹下

```bash
streamlit run orange_classifier_web.py
```

## 效果展示

1. 主页
   
   ![主页.png](E:\Project\ComputerVision\Plant_foliar_disease_and_pest_detection\主页.png)

2. 训练过程
   
   ![训练过程.png](E:\Project\ComputerVision\Plant_foliar_disease_and_pest_detection\训练过程.png)

3. 模型预测
   
   ![模型预测.png](E:\Project\ComputerVision\Plant_foliar_disease_and_pest_detection\模型预测.png)

4. 训练指标
   
   ![训练指标.png](E:\Project\ComputerVision\Plant_foliar_disease_and_pest_detection\训练指标.png)
