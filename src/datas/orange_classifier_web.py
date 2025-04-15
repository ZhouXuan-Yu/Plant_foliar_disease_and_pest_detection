import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
import sys
import pandas as pd
import matplotlib as mpl
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入分类器模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datas.orange_classifier_fixed import OrangeClassifier, Config

# 设置页面配置
st.set_page_config(
    page_title="橙子叶片分类器",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #FF8C00;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-title {
        font-size: 1.8rem;
        color: #FF6347;
        margin-bottom: 15px;
    }
    .section-title {
        font-size: 1.5rem;
        color: #FF4500;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .info-text {
        font-size: 1.1rem;
    }
    .image-caption {
        text-align: center;
        font-style: italic;
        margin-top: 5px;
    }
    .stProgress .st-bo {
        background-color: #FF8C00;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .high-confidence {
        background-color: rgba(50, 205, 50, 0.2);
        border: 1px solid #32CD32;
    }
    .medium-confidence {
        background-color: rgba(255, 165, 0, 0.2);
        border: 1px solid #FFA500;
    }
    .low-confidence {
        background-color: rgba(255, 69, 0, 0.2);
        border: 1px solid #FF4500;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF8C00;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# 显示标题
st.markdown("<h1 class='main-title'>🍊 橙子叶片病害分类系统</h1>", unsafe_allow_html=True)

# 侧边栏
st.sidebar.markdown("## 导航")
page = st.sidebar.radio("选择功能", ["主页", "训练过程", "训练指标", "模型预测"])

# 设置输出路径常量
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.png")
HIRES_HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history_hires.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
SAMPLES_PATH = os.path.join(OUTPUT_DIR, "training_samples.png")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")

# 将路径常量添加到配置中
config = Config()
config.output_dir = OUTPUT_DIR
config.history_path = HISTORY_PATH
config.hires_history_path = HIRES_HISTORY_PATH
config.confusion_matrix_path = CONFUSION_MATRIX_PATH
config.samples_path = SAMPLES_PATH
config.metrics_path = METRICS_PATH

# 加载模型
@st.cache_resource
def load_model(model_path, num_classes=3):
    """加载预训练模型"""
    try:
        model = OrangeClassifier(num_classes=num_classes)
        if not os.path.exists(model_path):
            st.error(f"找不到模型文件: {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"加载模型出错: {str(e)}")
        return None

# 图像预处理
def preprocess_image(image, size=224):
    """预处理上传的图像"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# 加载训练指标
def load_metrics():
    """加载训练和评估指标"""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"加载指标文件出错: {str(e)}")
            return None
    else:
        st.warning(f"找不到指标文件: {METRICS_PATH}")
        return None

# 显示训练历史
def display_training_history():
    st.markdown("<h2 class='sub-title'>📈 训练过程可视化</h2>", unsafe_allow_html=True)
    
    # 尝试加载已经生成的训练历史图
    if os.path.exists(HIRES_HISTORY_PATH):
        st.image(HIRES_HISTORY_PATH, caption="训练和验证的损失与准确率", use_container_width=True)
    elif os.path.exists(HISTORY_PATH):
        st.image(HISTORY_PATH, caption="训练和验证的损失与准确率", use_container_width=True)
    else:
        st.warning("找不到训练历史图表。请先运行模型训练。")
        st.info("训练历史图表应保存在以下位置：")
        st.code(HISTORY_PATH)
    
    # 显示混淆矩阵
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.markdown("<h3 class='section-title'>混淆矩阵</h3>", unsafe_allow_html=True)
        st.image(CONFUSION_MATRIX_PATH, caption="测试集上的混淆矩阵", use_container_width=True)
    else:
        st.warning("找不到混淆矩阵图表。请先运行模型评估。")
    
    # 显示样本图像
    if os.path.exists(SAMPLES_PATH):
        st.markdown("<h3 class='section-title'>训练样本示例</h3>", unsafe_allow_html=True)
        st.image(SAMPLES_PATH, caption="训练样本示例", use_container_width=True)
    else:
        st.warning("找不到训练样本示例图片。请先运行模型训练。")

# 显示训练指标
def display_training_metrics():
    st.markdown("<h2 class='sub-title'>📊 模型评估指标</h2>", unsafe_allow_html=True)
    
    metrics = load_metrics()
    if not metrics:
        st.warning("找不到评估指标数据。请先运行模型评估。")
        st.info("您可以通过运行以下命令进行模型评估：")
        st.code("python orange_classifier_test.py")
        return
    
    # 显示整体评估指标
    st.markdown("<h3 class='section-title'>整体评估指标</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('accuracy', 0):.4f}</div>
            <div class="metric-label">准确率 (Accuracy)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('precision', 0):.4f}</div>
            <div class="metric-label">精确率 (Precision)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('recall', 0):.4f}</div>
            <div class="metric-label">召回率 (Recall)</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('f1', 0):.4f}</div>
            <div class="metric-label">F1分数 (F1 Score)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('kappa', 0):.4f}</div>
            <div class="metric-label">Kappa系数 (Cohen's Kappa)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics.get('hamming_loss', 0):.4f}</div>
            <div class="metric-label">汉明损失 (Hamming Loss)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示每个类别的评估指标
    if 'class_metrics' in metrics:
        st.markdown("<h3 class='section-title'>各类别评估指标</h3>", unsafe_allow_html=True)
        
        class_data = []
        for cls_name, cls_metrics in metrics['class_metrics'].items():
            class_data.append({
                "类别": cls_name,
                "样本数": cls_metrics.get('support', 0),
                "正确预测": cls_metrics.get('correct', 0),
                "准确率": cls_metrics.get('accuracy', 0),
                "精确率": cls_metrics.get('precision', 0),
                "召回率": cls_metrics.get('recall', 0),
                "F1分数": cls_metrics.get('f1', 0)
            })
        
        # 创建数据框并显示
        df = pd.DataFrame(class_data)
        st.dataframe(df, use_container_width=True)
    
    # 显示训练配置
    if 'config' in metrics:
        with st.expander("训练配置详情"):
            config_data = metrics['config']
            for key, value in config_data.items():
                st.write(f"**{key}:** {value}")

# 预测图像
def predict_image(model, image, class_names):
    """使用模型预测图像类别"""
    try:
        # 预处理图像
        input_tensor = preprocess_image(image)
        
        # 记录开始时间
        start_time = time.time()
        
        # 进行预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # 计算预测时间
        pred_time = time.time() - start_time
        
        # 获取预测类别和概率
        pred_probs, pred_classes = torch.topk(probabilities, len(class_names))
        
        # 转换为Python列表
        pred_probs = pred_probs.cpu().numpy()
        pred_classes = pred_classes.cpu().numpy()
        
        results = []
        for i in range(len(class_names)):
            results.append({
                'class': class_names[pred_classes[i]],
                'probability': float(pred_probs[i])
            })
        
        return results, pred_time
    except Exception as e:
        st.error(f"预测出错: {str(e)}")
        return None, 0

# 显示预测结果
def display_prediction(results, uploaded_image, pred_time):
    # 显示原始图像
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_image, caption="上传的图像", use_container_width=True)
    
    with col2:
        st.markdown("<h3 class='section-title'>预测结果</h3>", unsafe_allow_html=True)
        
        # 显示预测结果
        for i, res in enumerate(results):
            confidence = res['probability']
            
            # 根据置信度分配不同的样式
            if confidence >= 0.7:
                box_class = "high-confidence"
                confidence_text = "高置信度"
            elif confidence >= 0.4:
                box_class = "medium-confidence"
                confidence_text = "中置信度"
            else:
                box_class = "low-confidence"
                confidence_text = "低置信度"
            
            # 显示预测结果
            st.markdown(f"""
            <div class='prediction-box {box_class}'>
                <p><strong>{i+1}. 类别:</strong> {res['class']}</p>
                <p><strong>置信度:</strong> {confidence:.2%} ({confidence_text})</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 显示结果解释
        if results and results[0]['probability'] >= 0.7:
            st.success(f"该图像很可能是 {results[0]['class']} 类别，置信度为 {results[0]['probability']:.2%}")
        elif results and results[0]['probability'] >= 0.4:
            st.warning(f"该图像可能是 {results[0]['class']} 类别，但置信度不高 ({results[0]['probability']:.2%})")
        else:
            st.error("无法确定图像类别，置信度过低")
    
    # 显示技术细节
    with st.expander("查看技术细节"):
        st.write("模型架构: ResNet18")
        st.write(f"图像尺寸: {uploaded_image.size}")
        st.write(f"预测耗时: {pred_time:.4f}秒")
        st.write("预处理步骤: 调整大小、标准化")
        
        # 显示所有类别的概率
        st.write("所有类别的预测概率:")
        probs_df = pd.DataFrame(results)
        probs_df.columns = ["类别", "概率"]
        st.dataframe(probs_df)

# 主页
def show_home():
    st.markdown("<h2 class='sub-title'>🧠 橙子叶片病害分类系统</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='info-text'>
    这个系统使用深度学习技术对橙子叶片进行分类，帮助识别不同类型的病害。系统基于ResNet18预训练模型，针对以下类别进行训练：
    </p>
    """, unsafe_allow_html=True)
    
    # 显示类别信息
    classes = ["健康 (healthy)", "黑斑病 (citrus canker)", "黑点病 (melanose)"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border:1px solid #32CD32; border-radius:5px; padding:10px; text-align:center;">
            <h3 style="color:#32CD32;">健康叶片</h3>
            <p>正常生长的橙子叶片，没有任何病害症状</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border:1px solid #FFA500; border-radius:5px; padding:10px; text-align:center;">
            <h3 style="color:#FFA500;">柑橘黄斑病</h3>
            <p>叶片上出现黄褐色斑点，边缘常有水渍状晕圈</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border:1px solid #FF4500; border-radius:5px; padding:10px; text-align:center;">
            <h3 style="color:#FF4500;">黑点病</h3>
            <p>叶片表面有小的黑色点状突起，质地粗糙</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-title'>系统功能</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='info-text'>
    1. <strong>训练过程</strong>：查看模型训练过程的损失和准确率变化曲线<br>
    2. <strong>训练指标</strong>：查看模型在测试集上的性能指标和评估结果<br>
    3. <strong>模型预测</strong>：上传橙子叶片图像，获取分类结果<br>
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-title'>使用说明</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <p class='info-text'>
    1. 在侧边栏选择您要使用的功能<br>
    2. 如需进行预测，请上传清晰的橙子叶片图像<br>
    3. 查看预测结果和置信度<br>
    </p>
    """, unsafe_allow_html=True)
    
    st.info("为获得最佳效果，请上传清晰的橙子叶片图像，背景简单，叶片占主要部分。")

# 模型预测
def show_prediction():
    st.markdown("<h2 class='sub-title'>🔍 模型预测</h2>", unsafe_allow_html=True)
    
    # 定义类别名称
    class_names = ["healthy", "citrus canker", "melanose"]
    class_labels = {
        "healthy": "健康",
        "citrus canker": "柑橘黄斑病",
        "melanose": "黑点病"
    }
    
    # 检查模型文件是否存在
    model_path = config.model_save_path
    
    if not os.path.exists(model_path):
        st.warning(f"找不到模型文件: {model_path}。请先运行训练脚本。")
        if st.button("运行训练"):
            st.info("正在训练模型，请稍候...")
            # 这里可以添加训练过程的进度显示
            st.warning("训练功能尚未实现，请先运行训练脚本: python orange_classifier_train.py")
        return
    
    # 加载模型
    model = load_model(model_path, num_classes=len(class_names))
    if model is None:
        return
    
    # 文件上传
    uploaded_file = st.file_uploader("上传橙子叶片图像", type=["jpg", "jpeg", "png"])
    
    # 添加示例图像选项
    use_example = st.checkbox("使用示例图像")
    
    if use_example:
        # 检查是否有示例图像目录
        example_dir = os.path.join(config.split_dir, "test")
        if os.path.exists(example_dir):
            # 获取示例图像列表
            example_classes = [d for d in os.listdir(example_dir) if os.path.isdir(os.path.join(example_dir, d))]
            if example_classes:
                selected_class = st.selectbox("选择示例类别", example_classes)
                class_dir = os.path.join(example_dir, selected_class)
                
                example_images = [f for f in os.listdir(class_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if example_images:
                    selected_image = st.selectbox("选择示例图像", example_images)
                    image_path = os.path.join(class_dir, selected_image)
                    
                    with st.spinner("正在处理图像..."):
                        # 打开图像
                        image = Image.open(image_path).convert('RGB')
                        
                        # 进行预测
                        results, pred_time = predict_image(model, image, class_names)
                        
                        if results:
                            # 修改结果显示中文类别名
                            for res in results:
                                if res['class'] in class_labels:
                                    res['class'] = f"{class_labels[res['class']]} ({res['class']})"
                            
                            # 显示预测结果
                            display_prediction(results, image, pred_time)
                else:
                    st.warning(f"所选类别 '{selected_class}' 中没有图像文件")
            else:
                st.warning("找不到示例类别")
        else:
            st.warning(f"找不到示例图像目录: {example_dir}")
    
    elif uploaded_file is not None:
        # 显示处理中状态
        with st.spinner("正在处理图像..."):
            # 打开图像
            image = Image.open(uploaded_file).convert('RGB')
            
            # 进行预测
            results, pred_time = predict_image(model, image, class_names)
            
            if results:
                # 修改结果显示中文类别名
                for res in results:
                    if res['class'] in class_labels:
                        res['class'] = f"{class_labels[res['class']]} ({res['class']})"
                
                # 显示预测结果
                display_prediction(results, image, pred_time)
    else:
        st.info("请上传一张橙子叶片图像进行预测或选择使用示例图像")

# 主程序
if __name__ == "__main__":
    if page == "主页":
        show_home()
    elif page == "训练过程":
        display_training_history()
    elif page == "训练指标":
        display_training_metrics()
    elif page == "模型预测":
        show_prediction()

    # 页脚
    st.markdown("---")
    st.markdown("<p style='text-align: center'>© 2023 橙子叶片病害分类系统 | 基于PyTorch和Streamlit构建</p>", unsafe_allow_html=True) 