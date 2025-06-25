try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from lib import newIcon, labelValidator

BB = QDialogButtonBox


class LabelDialog(QDialog):

    def __init__(self, text="Enter object label", parent=None, listItem=None):
        super(LabelDialog, self).__init__(parent)
        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        self.edit.installEventFilter(self)  # 安装事件过滤器

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        # if listItem is not None and len(listItem) > 0:
        #     self.listWidget = QListWidget(self)
        #     for item in listItem:
        #         self.listWidget.addItem(item)
        #     self.listWidget.itemDoubleClicked.connect(self.listItemClick)
        #     layout.addWidget(self.listWidget)

        # self.setLayout(layout)

        # 存储标签列表
        self.label_items = listItem if listItem else []

        if self.label_items:
            self.listWidget = QListWidget(self)
            for index, item in enumerate(self.label_items):
                # 在每个标签前添加数字索引
                item_text = f"{index + 1}: {item}"
                list_item = QListWidgetItem(item_text)
                self.listWidget.addItem(list_item)
            self.listWidget.itemDoubleClicked.connect(self.listItemClick)
            layout.addWidget(self.listWidget)

        self.setLayout(layout)

    def validate(self):
        try:
            if self.edit.text().trimmed():
                self.accept()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            if self.edit.text().strip():
                self.accept()

    def postProcess(self):
        try:
            self.edit.setText(self.edit.text().trimmed())
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            self.edit.setText(self.edit.text())

    def popUp(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.edit.text() if self.exec_() else None

    def listItemClick_old(self, tQListWidgetItem):
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.edit.setText(text)
        self.validate()

    def listItemClick(self, tQListWidgetItem):
        # 移除数字索引，获取实际的标签文本
        text = tQListWidgetItem.text()
        if ': ' in text:
            text = text.split(': ', 1)[1]
        self.edit.setText(text)
        self.validate()

    def eventFilter(self, obj, event):
        if obj == self.edit and event.type() == QEvent.KeyPress:
            key = event.key()
            if Qt.Key_1 <= key <= Qt.Key_9:
                index = key - Qt.Key_1  # 索引从 0 开始
                if index < len(self.label_items):
                    label = self.label_items[index]
                    self.edit.setText(label)
                    self.validate()
                    return True  # 表示事件已处理
            elif key == Qt.Key_0:
                index = 9  # '0' 键对应索引 9
                if index < len(self.label_items):
                    label = self.label_items[index]
                    self.edit.setText(label)
                    self.validate()
                    return True
        # 对于其他按键，调用父类的事件处理
        return super(LabelDialog, self).eventFilter(obj, event)