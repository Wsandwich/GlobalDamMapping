import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QFileDialog, QAction)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize


class AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Image Annotation Tool")
        self.setGeometry(100, 100, 1200, 600)

        # 计算每个标签的固定正方形大小（宽度等于窗口宽度的三分之一）
        square_size = self.width() // 3

        # 创建三个图像标签用于显示三幅图像，并设置固定的正方形大小
        self.image_labels = [QLabel(self), QLabel(self), QLabel(self)]

        for label in self.image_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black;")  # 设置边框方便查看布局
            label.setFixedSize(QSize(square_size, square_size))  # 设置为固定正方形大小

        # 设置布局
        image_layout = QHBoxLayout()
        for label in self.image_labels:
            image_layout.addWidget(label)
            image_layout.setStretch(image_layout.count() - 1, 1)  # 设置标签占据布局等分空间

        # 中心窗口和布局
        central_widget = QWidget()
        central_widget.setLayout(image_layout)
        self.setCentralWidget(central_widget)

        # 创建菜单栏
        self.create_menu()

    def create_menu(self):
        # 创建菜单栏和动作
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        open_action = QAction("Open Images", self)
        open_action.triggered.connect(self.open_images)
        file_menu.addAction(open_action)

        save_action = QAction("Save Annotations", self)
        save_action.triggered.connect(self.save_annotations)
        file_menu.addAction(save_action)

    def open_images(self):
        # 打开多个图像文件，并在标签中显示
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Image Files (*.png *.jpg *.bmp)")

        for i, file_path in enumerate(file_paths[:3]):  # 仅加载前三幅图像
            pixmap = QPixmap(file_path)
            self.image_labels[i].setPixmap(
                pixmap.scaled(self.image_labels[i].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_annotations(self):
        # 保存标注文件的占位符函数
        print("Annotations saved (to be implemented)")


# 主程序入口
app = QApplication(sys.argv)
window = AnnotationTool()
window.show()
sys.exit(app.exec_())
