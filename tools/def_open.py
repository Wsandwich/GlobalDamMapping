class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        # ... [初始化其他成员变量和UI组件] ...

        # Actions
        action = partial(newAction, self)
        # ... [定义其他动作] ...

        # 定义新的 "Next Labeled Image" 动作
        openNextLabeledImg = action('&Next Labeled Image', self.openNextLabeledImg,
                                   'Ctrl+Shift+N', 'next-labeled', u'Open Next Labeled Image')

        # 定义新的 "Previous Labeled Image" 动作
        openPrevLabeledImg = action('&Previous Labeled Image', self.openPrevLabeledImg,
                                   'Ctrl+Shift+P', 'prev-labeled', u'Open Previous Labeled Image')

        # 将新的动作添加到 File 菜单
        addActions(self.menus.file, (openNextLabeledImg, openPrevLabeledImg))

        # 或者，如果你希望将其添加到工具栏，可以这样做：
        # self.tools.addAction(openNextLabeledImg)
        # self.tools.addAction(openPrevLabeledImg)

        # Store actions for further handling.
        self.actions = struct(
            # ... [其他动作] ...
            openNextLabeledImg=openNextLabeledImg,
            openPrevLabeledImg=openPrevLabeledImg,
            # ... [其他动作] ...
        )

        # ... [其余初始化代码] ...

    # ... [其他方法保持不变] ...

    def openNextLabeledImg(self, _value=False):
        """
        跳转到下一张有标签的图像。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            # 如果当前没有打开的文件，尝试加载第一个有标签的文件
            for i in range(len(self.mImgList)):
                if self.has_labels(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到有标签的图像。')
            return

        try:
            # 获取当前组的索引
            current_group_index = self.mImgList.index(
                next(
                    (group for group in self.mImgList if self.filePath in group[1]),
                    None
                )
            )
        except StopIteration:
            current_group_index = -1

        # 从当前索引的下一个开始查找
        for i in range(current_group_index + 1, len(self.mImgList)):
            if self.has_labels(i):
                self.loadFile(i)
                return

        # 如果到达末尾，还未找到，提示用户
        self.statusBar().showMessage(u'已到达最后一张有标签的图像。')

    def openPrevLabeledImg(self, _value=False):
        """
        跳转到上一张有标签的图像。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            # 如果当前没有打开的文件，尝试加载最后一个有标签的文件
            for i in reversed(range(len(self.mImgList))):
                if self.has_labels(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到有标签的图像。')
            return

        try:
            # 获取当前组的索引
            current_group_index = self.mImgList.index(
                next(
                    (group for group in self.mImgList if self.filePath in group[1]),
                    None
                )
            )
        except StopIteration:
            current_group_index = -1

        # 从当前索引的前一个开始查找
        for i in reversed(range(0, current_group_index)):
            if self.has_labels(i):
                self.loadFile(i)
                return

        # 如果到达开始，还未找到，提示用户
        self.statusBar().showMessage(u'已到达第一张有标签的图像。')

    def has_labels(self, group_index):
        """
        判断指定组是否有标签。
        """
        prefix, image_paths = self.mImgList[group_index]
        main_image_path = image_paths[0]
        annotation_filename = os.path.splitext(os.path.basename(main_image_path))[0] + XML_EXT
        annotation_path = os.path.join(self.defaultSaveDir, annotation_filename)
        if not os.path.exists(annotation_path):
            return False
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        return True
            return False
        except:
            return False

    # ... [其余方法保持不变] ...
