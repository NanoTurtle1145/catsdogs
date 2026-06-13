import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QWidget, QFileDialog, QMessageBox, QFrame, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSignal

# Matplotlib 后端
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==========================================
# 1. 后端逻辑 (基本不变，增加了图片预处理适配)
# ==========================================
class CatDogPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ['cat', 'dog'] # 默认值
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = models.resnet18(pretrained=False)

            # 兼容直接存 state_dict 或存字典
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
                if 'classes' in checkpoint: self.classes = checkpoint['classes']
            else:
                state_dict = checkpoint

            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.classes))
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            return True, f"模型加载成功! 类别: {self.classes}"
        except Exception as e:
            return False, f"加载失败: {str(e)}"

    def predict(self, image_path):
        if not self.model: return None
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
            return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        except Exception as e:
            print(e)
            return None

# ==========================================
# 2. 自定义美化组件
# ==========================================
class PredictionBar(QFrame):
    """自定义的概率条组件"""
    def __init__(self, label_text, color):
        super().__init__()
        self.color = color
        self.layout = QHBoxLayout(self)

        self.label = QLabel(label_text.capitalize())
        self.label.setFixedWidth(50)
        self.label.setStyleSheet(f"color: {color}; font-weight: bold;")

        self.bar_bg = QFrame()
        self.bar_bg.setFixedHeight(24)
        self.bar_bg.setStyleSheet(f"background-color: #444; border-radius: 12px;")
        self.bar_fill = QFrame(self.bar_bg)
        self.bar_fill.setGeometry(0, 0, 0, 24) # 初始宽度为0
        self.bar_fill.setStyleSheet(f"background-color: {color}; border-radius: 12px;")

        self.value_label = QLabel("0%")
        self.value_label.setFixedWidth(60)
        self.value_label.setStyleSheet("color: white;")

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.bar_bg, 1)
        self.layout.addWidget(self.value_label)

    def set_value(self, value):
        # value: 0.0 - 1.0
        width = int(self.bar_bg.width() * value)
        self.bar_fill.setGeometry(0, 0, width, 24)
        self.value_label.setText(f"{value*100:.2f}%")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = CatDogPredictor()
        self.current_img_path = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🐱 猫狗分类器 Pro 🐶")
        self.setGeometry(100, 100, 1000, 650)

        # --- 设置全局样式 (QSS) ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QLabel {
                font-family: 'Segoe UI', 'Microsoft YaHei';
                font-size: 14px;
            }
            QPushButton {
                background-color: #3c3f41;
                border: 1px solid #555;
                padding: 10px;
                border-radius: 8px;
                font-size: 14px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #505354;
                border: 1px solid #007acc;
            }
            QPushButton:pressed {
                background-color: #007acc;
            }
            QPushButton:disabled {
                background-color: #3c3f41;
                color: #666;
            }
            QFrame#card {
                background-color: #3c3f41;
                border-radius: 10px;
                padding: 15px;
            }
        """)

        # --- 中心部件 ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # --- 左侧控制面板 ---
        left_panel = QFrame(objectName="card")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        self.btn_load_model = QPushButton("📂 加载模型")
        self.btn_load_image = QPushButton("🖼️ 选择图片")
        self.btn_predict = QPushButton("🚀 开始预测")
        self.btn_predict.setEnabled(False)

        self.lbl_status = QLabel("请先加载模型 (.pth)")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #aaa; padding: 10px;")

        # 预测结果展示区域
        result_frame = QFrame(objectName="card")
        result_layout = QVBoxLayout(result_frame)
        self.cat_bar = PredictionBar("Cat", "#FF9800") # 橙色
        self.dog_bar = PredictionBar("Dog", "#2196F3") # 蓝色
        result_layout.addWidget(QLabel("预测结果:"))
        result_layout.addWidget(self.cat_bar)
        result_layout.addWidget(self.dog_bar)

        left_layout.addWidget(self.btn_load_model)
        left_layout.addWidget(self.btn_load_image)
        left_layout.addWidget(self.btn_predict)
        left_layout.addWidget(self.lbl_status)
        left_layout.addStretch(1)
        left_layout.addWidget(result_frame)

        # --- 右侧显示面板 ---
        right_panel = QFrame(objectName="card")
        right_layout = QVBoxLayout(right_panel)

        # 图片显示
        self.lbl_image = QLabel("图片预览区域")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumSize(400, 400)
        self.lbl_image.setStyleSheet("border: 2px dashed #555; border-radius: 10px; color: #666;")

        # Matplotlib 图表
        self.canvas = FigureCanvas(Figure(figsize=(5, 3), facecolor='#3c3f41'))
        self.ax = self.canvas.figure.subplots()
        self.ax.set_facecolor('#3c3f41')
        self.ax.tick_params(colors='white')
        self.ax.spines[:].set_color('#555')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_title("概率分布图")
        self.ax.set_ylabel("Probability")
        self.ax.set_xlabel("Class")

        right_layout.addWidget(self.lbl_image, 1)
        right_layout.addWidget(self.canvas, 1)

        # --- 加入主布局 ---
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)

        # --- 信号连接 ---
        self.btn_load_model.clicked.connect(self.load_model)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_predict.clicked.connect(self.run_prediction)

    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PTH Files (*.pth)")
        if fname:
            success, msg = self.predictor.load_model(fname)
            if success:
                self.lbl_status.setText(msg)
                self.btn_predict.setEnabled(True)
            else:
                QMessageBox.critical(self, "错误", msg)

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.current_img_path = fname
            pixmap = QPixmap(fname)
            # 缩放并保持比例
            scaled = pixmap.scaled(self.lbl_image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
            self.lbl_status.setText(f"已加载: {fname.split('/')[-1]}")

    def run_prediction(self):
        if not self.current_img_path: return

        self.lbl_status.setText("正在推理...")
        QApplication.processEvents()

        results = self.predictor.predict(self.current_img_path)

        if results:
            # 更新自定义进度条
            self.cat_bar.set_value(results.get('cat', 0))
            self.dog_bar.set_value(results.get('dog', 0))

            # 更新 Matplotlib 图
            self.ax.clear()
            classes = list(results.keys())
            probs = list(results.values())
            colors = ['#FF9800' if c == 'cat' else '#2196F3' for c in classes]

            bars = self.ax.bar(classes, probs, color=colors)
            self.ax.set_ylim(0, 1)
            self.ax.set_title("概率分布图")
            for bar in bars:
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height+0.02,
                             f'{height:.2%}', ha='center', color='white')
            self.canvas.draw()

            max_class = max(results, key=results.get)
            self.lbl_status.setText(f"预测结果: {max_class.upper()} ({results[max_class]*100:.2f}%)")
        else:
            self.lbl_status.setText("预测失败，请检查模型或图片。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
