#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os.path
import re
import sys
import subprocess
import shutil
from functools import partial
from collections import defaultdict

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import resources
# Add internal libs
dir_name = os.path.abspath(os.path.dirname(__file__))
libs_path = os.path.join(dir_name, 'libs')
sys.path.insert(0, libs_path)
from lib import struct, newAction, newIcon, addActions, fmtShortcut
from shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from canvas import Canvas
from zoomWidget import ZoomWidget
from labelDialog import LabelDialog
from colorDialog import ColorDialog
from labelFile import LabelFile, LabelFileError
from toolBar import ToolBar
from pascal_voc_io import PascalVocReader, DotaReader
from pascal_voc_io import XML_EXT
from ustr import ustr



__appname__ = 'roLabelImg'

# Utility functions and classes.


def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))


def util_qt_strlistclass():
    return QStringList if have_qstring() else list


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


# PyQt5: TypeError: unhashable type: 'QListWidgetItem'
class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        super(HashableQListWidgetItem, self).__init__(*args)

    def __hash__(self):
        return hash(id(self))


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        # Save as Pascal voc xml
        self.defaultSaveDir = None
        self.usingPascalVocFormat = True
        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self.isEnableCreate = True
        self.isEnableCreateRo = True

        # Enble auto saving if pressing next
        self.autoSaving = True
        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = "firefox"
        self.screencast = "https://youtu.be/7D5lvol_QRA"
        # For a demo of original labelImg, please see "https://youtu.be/p0nR2YsCY_U"

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)
        
        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)
        
        # Create a widget for using default label
        self.useDefautLabelCheckbox = QCheckBox(u'Use default label')
        self.useDefautLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefautLabelQHBoxLayout = QHBoxLayout()       
        useDefautLabelQHBoxLayout.addWidget(self.useDefautLabelCheckbox)
        useDefautLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefautLabelContainer = QWidget()
        useDefautLabelContainer.setLayout(useDefautLabelQHBoxLayout)


        # After this section, add the coordinates display:
        # Add coordinates display and copy button
        self.coordsLabel = QLabel("")
        self.coordsLabel.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.coordsLabel.setStyleSheet("padding: 5px;")
        self.copyCoordButton = QPushButton("Copy Coordinates")
        self.copyCoordButton.clicked.connect(self.copyCoordinatesToClipboard)
        self.copyCoordButton.setFixedHeight(30)

        # Create layout for coordinates components
        coordsLayout = QHBoxLayout()
        coordsLayout.addWidget(self.coordsLabel, 1)  # 1 is the stretch factor
        coordsLayout.addWidget(self.copyCoordButton)
        coordsContainer = QWidget()
        coordsContainer.setLayout(coordsLayout)

        # Add the coordinates container to the listLayout after useDefautLabelContainer
        listLayout.addWidget(useDefautLabelContainer)
        listLayout.addWidget(coordsContainer)


        # Add auto-zoom toggle checkbox
        self.autoZoomCheckbox = QCheckBox(u'Auto Zoom to Selection')
        self.autoZoomCheckbox.setChecked(False)  # Default to off
        self.autoZoomCheckbox.stateChanged.connect(self.toggleAutoZoom)
        
        # Add the checkbox to the appropriate layout
        # For example, if you want to add it near other label-related controls:
        useDefautLabelQHBoxLayout.addWidget(self.autoZoomCheckbox)
        
        # Or if you prefer to add it to the list layout below other controls:
        # listLayout.addWidget(self.autoZoomCheckbox)
        
        # Initialize auto-zoom flag
        self.autoZoomEnabled = False

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(u'difficult')
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)

        # Add progress tracking label with motivational messages
        self.progressLabel = QLabel()
        self.progressLabel.setAlignment(Qt.AlignCenter)
        self.progressLabel.setStyleSheet("font-weight: bold; color: #2C5F2D; padding: 5px; background-color: #f0f9e8; border-radius: 4px;")
        self.progressLabel.setWordWrap(True)
        
        # Add the progress label to the layout below the difficulty button
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(self.progressLabel)
        
        # Initialize progress tracking variables
        self.totalImageGroups = 0
        self.completedImageGroups = 0
        self.lastCheckedGroups = {}  # To track which groups we've already checked


        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add some of widgets to listLayout 
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(useDefautLabelContainer)

        #Add lable in different year
        self.labelyears = QListWidget()
        self.labelyears.setFixedSize(400, 100)
        listLayout.addWidget(self.labelyears)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)


        # 添加标志位，用于忽略 labelList 的选择变化
        self.ignore_label_selection = False

        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)

        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)

        self.dock = QDockWidget(u'Box Labels', self)
        self.dock.setObjectName(u'Label')
        self.dock.setWidget(labelListContainer)

        # Tzutalin 20160906 : Add file list and dock to move faster
        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(u'File List', self)
        self.filedock.setObjectName(u'File')
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        # self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas()
        self.canvas_1 = Canvas()
        self.canvas_2 = Canvas()

        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas_1.zoomRequest.connect(self.zoomRequest)  # 可能需要不同的处理
        self.canvas_2.zoomRequest.connect(self.zoomRequest)  # 或者相同的处理但参数不同

        self.zoomWidget.setMinimum(10)   # 最小缩放级别 10%
        self.zoomWidget.setMaximum(500)  # 最大缩放级别 500%
        self.zoomWidget.setValue(100)    # 初始缩放级别 100%
        self.zoom_level = 100            # 同步记录当前缩放级别
        self.current_center = None

        self.scroll = QScrollArea()  # 使用 self.scroll
        self.scroll_1 = QScrollArea()
        self.scroll_2 = QScrollArea()

        # 连接滚动条的 valueChanged 信号
        self.scroll.horizontalScrollBar().valueChanged.connect(self.update_scroll)
        self.scroll.verticalScrollBar().valueChanged.connect(self.update_scroll)
        self.scroll_1.horizontalScrollBar().valueChanged.connect(self.update_scroll_1)
        self.scroll_1.verticalScrollBar().valueChanged.connect(self.update_scroll_1)
        self.scroll_2.horizontalScrollBar().valueChanged.connect(self.update_scroll_2)
        self.scroll_2.verticalScrollBar().valueChanged.connect(self.update_scroll_2)

        self.scroll.setWidget(self.canvas)
        self.scroll_1.setWidget(self.canvas_1)
        self.scroll_2.setWidget(self.canvas_2)
        self.scroll.setWidgetResizable(True)
        self.scroll_1.setWidgetResizable(True)
        self.scroll_2.setWidgetResizable(True)

        self.scrollBars = {
            Qt.Vertical: self.scroll.verticalScrollBar(),
            Qt.Horizontal: self.scroll.horizontalScrollBar()
        }
        self.scrollBars_1 = {
            Qt.Vertical: self.scroll_1.verticalScrollBar(),
            Qt.Horizontal: self.scroll_1.horizontalScrollBar()
        }
        self.scrollBars_2 = {
            Qt.Vertical: self.scroll_2.verticalScrollBar(),
            Qt.Horizontal: self.scroll_2.horizontalScrollBar()
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.canvas_1.scrollRequest.connect(self.scrollRequest)
        self.canvas_2.scrollRequest.connect(self.scrollRequest)

        central_widget = QWidget()
        central_widget.setFixedSize(1800, 1000)

        self.main_layout = QVBoxLayout(central_widget)
        up_layout = QHBoxLayout()
        low_layout = QGridLayout()

        # Create a label to display annotation counts
        self.annotationCountLabel = QLabel("Annotations: 2010: 0 | 2015: 0 | 2020: 0 | Total: 0")
        self.annotationCountLabel.setAlignment(Qt.AlignCenter)
        self.annotationCountLabel.setStyleSheet("background-color: rgba(255, 255, 255, 180); padding: 3px;")

        # Add the label to an appropriate layout
        # For example, adding it to the low_layout (below the image canvases)
        low_layout.addWidget(self.annotationCountLabel, 0, 0, 1, 3)  # Span across 3 columns



        up_layout.addWidget(self.scroll)
        up_layout.addWidget(self.scroll_1)
        up_layout.addWidget(self.scroll_2)

        shortcuts = [
            (self.add_2010, "1"),
            (self.add_2015, "2"),
            (self.add_2020, "3"),
            (self.delete_2010, "4"),
            (self.delete_2015, "5"),
            (self.delete_2020, "6"),
            (self.delete_all_label, "7"),
            (self.delete_and_next, "8"),
            (self.embankment, "/"),
            (self.Barrage, "*"),
            (self.gravity, "-"),
            (self.arch, "+")
        ]


        for action, key in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)  # self 为窗口类实例
            shortcut.activated.connect(action)  # 绑定快捷键触发相应的操作

        self.main_layout.addLayout(up_layout)
        self.main_layout.addLayout(low_layout)

        self.setCentralWidget(central_widget)  # 设置 self.scroll 为中央窗口部件

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas_1.newShape.connect(self.newShape_1)
        self.canvas_1.shapeMoved.connect(self.setDirty)
        self.canvas_2.newShape.connect(self.newShape_2)
        self.canvas_2.shapeMoved.connect(self.setDirty)

        # self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.selectionChanged.connect(self.on_canvas_selection_changed)
        self.canvas_1.selectionChanged.connect(self.on_canvas_selection_changed_1)
        self.canvas_2.selectionChanged.connect(self.on_canvas_selection_changed_2)

        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.status.connect(self.status)
        self.canvas_1.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas_1.status.connect(self.status)
        self.canvas_2.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas_2.status.connect(self.status)

        self.canvas.hideNRect.connect(self.enableCreate)
        self.canvas.hideRRect.connect(self.enableCreateRo)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        # Tzutalin 20160906 : Add file list and dock to move faster
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.dockFeatures = QDockWidget.DockWidgetClosable\
            | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)
        self.filedock.setFeatures(self.filedock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)

        # Initialize small annotation threshold values
        self.small_annotation_width_threshold = 1.0
        self.small_annotation_height_threshold = 1.0

        # Add actions for small annotation detection
        setSmallAnnotationThresholdsAction = action('&Set Small Annotation Thresholds', 
                                                self.setSmallAnnotationThresholds,
                                                'Ctrl+T', 'small-thresholds', 
                                                u'Set thresholds for small annotation detection')

        jumpToSmallAnnotationsAction = action('&Jump to Small Annotations', 
                                            lambda: self.jumpToSmallAnnotations(
                                                threshold_width=self.small_annotation_width_threshold, 
                                                threshold_height=self.small_annotation_height_threshold),
                                            'Ctrl+V', 'jump-small', 
                                            u'Jump to next image group with small annotations')

        jumpToSmallAnnotationsBackwardAction = action('&Jump to Previous Small Annotations', 
                                                    lambda: self.jumpToSmallAnnotations(
                                                        backward=True,
                                                        threshold_width=self.small_annotation_width_threshold, 
                                                        threshold_height=self.small_annotation_height_threshold),
                                                    'Ctrl+Shift+V', 'jump-small-back', 
                                                    u'Jump to previous image group with small annotations')


        # 下一个有标签的图片
        openNextLabeledImg = action('&Next Labeled Image', self.openNextLabeledImg,
                                    'Shift+D', 'next-labeled', u'Open Next Labeled Image')

        # 上一个有标签的图片
        openPrevLabeledImg = action('&Previous Labeled Image', self.openPrevLabeledImg,
                                    'Shift+A', 'prev-labeled', u'Open Previous Labeled Image')

        # 下一个标签三年有变化的图片
        openNextChangedLabeledImg = action('&Next Changed Labeled Image', self.openNextChangedLabeledImg,
                                    'Ctrl+D', 'next-changed-labeled', u'Open Next Changed Labeled Image')

        # 上一个标签三年有变化的图片
        openPrevChangedLabeledImg = action('&Previous Changed Labeled Image', self.openPrevChangedLabeledImg,
                                    'Ctrl+A', 'prev-changed-labeled', u'Open Previous Changed Labeled Image')


        quit = action('&Quit', self.close,
                      'Ctrl+Q', 'quit', u'Quit application')

        open = action('&Open', self.openFile,
                      'Ctrl+O', 'open', u'Open image or label file')

        opendir = action('&Open Dir', self.openDir,
                         'Ctrl+u', 'open', u'Open Dir')

        changeSavedir = action('&Change default saved Annotation dir', self.changeSavedir,
                               'Ctrl+r', 'open', u'Change default saved Annotation dir')

        openAnnotation = action('&Open Annotation', self.openAnnotation,
                                'Ctrl+Shift+O', 'openAnnotation', u'Open Annotation')

        openNextImg = action('&Next Image', self.openNextImg,
                             'D', 'next', u'Open Next')

        openPrevImg = action('&Prev Image', self.openPrevImg,
                             'A', 'prev', u'Open Prev')

        verify = action('&Verify Image', self.verifyImg,
                        'space', 'verify', u'Verify Image')

        save = action('&Save', self.saveFile,
                      'Ctrl+S', 'save', u'Save labels to file', enabled=False)
        saveAs = action('&Save As', self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', u'Save labels to a different file',
                        enabled=False)
        close = action('&Close', self.closeFile,
                       'Ctrl+O', 'close', u'Close current file')
        color1 = action('Box &Line Color', self.chooseColor1,
                        'Ctrl+L', 'color_line', u'Choose Box line color')
        color2 = action('Box &Fill Color', self.chooseColor2,
                        'Ctrl+Shift+L', 'color', u'Choose Box fill color')

        createMode = action('Create\nRectBox', self.setCreateMode,
                            'Ctrl+N', 'new', u'Start drawing Boxs', enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action('Create\nRectBox', self.createShape,
                        'B', 'new', u'Draw a new Box', enabled=False)

        createRo = action('Create\nRotatedRBox', self.createRoShape,
                        'E', 'newRo', u'Draw a new RotatedRBox', enabled=False)

        delete = action('Delete\nRectBox', self.deleteSelectedShape,
                        'Delete', 'delete', u'Delete', enabled=False)
        copy = action('&Duplicate\nRectBox', self.copySelectedShape,
                      'Alt+D', 'copy', u'Create a duplicate of the selected Box',
                      enabled=False)

        advancedMode = action('&Advanced Mode', self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', u'Switch to advanced mode',
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', u'Hide all Boxs',
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Alt+A', 'hide', u'Show all Boxs',
                         enabled=False)

        help = action('&Tutorial', self.tutorial, 'Ctrl+T', 'help',
                      u'Show demos')
        
        next_three_img = action('&Next', self.Next, '', u'Next 3 img')
        back_three_img = action('&Back', self.Back, '', u'Back 3 img')

        # Add the shortcut keys for jumping forward
        jumpToDamAction = action('&Jump to Next Dam Label', self.jumpToDamLabel,
                            'Shift+E', 'jump-to-dam', u'Jump to next image labeled with gravity, arch, or barrage dams')
                            
        jumpToUnlabeledAction = action('&Jump to Next Unlabeled', self.jumpToUnlabeledGroup,
                                    'Shift+C', 'jump-to-unlabeled', u'Jump to next image group with no labels in all three years')

        # Add the shortcut keys for jumping backward
        jumpToDamBackwardAction = action('&Jump to Previous Dam Label', self.jumpToDamLabelBackward,
                                    'Shift+Q', 'jump-to-dam-back', u'Jump to previous image labeled with gravity, arch, or barrage dams')
                            
        jumpToUnlabeledBackwardAction = action('&Jump to Previous Unlabeled', self.jumpToUnlabeledGroupBackward,
                                            'Shift+Z', 'jump-to-unlabeled-back', u'Jump to previous image group with no labels in all three years')
        # Add this in the action definitions section
        setOutputDirsAction = action('&Set Output Directories', self.setOutputDirs,
                            'Ctrl+Shift+O', 'output-dirs', u'Set output directories for image and label files')
                            
        copyToOutputAction = action('&Copy to Output', self.copyToOutputDirs,
                            'P', 'copy-output', u'Copy current image group and labels to output directories')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In', partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', u'Increase zoom level', enabled=False)
        zoomOut = action('&Zoom Out', partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', u'Decrease zoom level', enabled=False)
        zoomOrg = action('&Original size', partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', u'Zoom to original size', enabled=False)
        fitWindow = action('&Fit Window', self.setFitWindow,
                           'Ctrl+F', 'fit-window', u'Zoom follows window size',
                           checkable=True, enabled=False)
        fitWidth = action('Fit &Width', self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', u'Zoom follows window width',
                          checkable=True, enabled=False)
        
        zoomToShapeAction = action('&Zoom to Selection', self.zoomToCurrentShape,
                          'Return', 'zoom-to-shape', u'Zoom to the currently selected annotation')

        # Create a second shortcut for the numpad Enter key
        numpadEnterShortcut = QShortcut(QKeySequence(Qt.Key_Enter), self)
        numpadEnterShortcut.activated.connect(self.zoomToCurrentShape)

        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action('&Edit Label', self.editLabel,
                      '0', 'edit', u'Modify the label of the selected Box',
                      enabled=False)
        
        self.editButton.setDefaultAction(edit)

        shapeLineColor = action('Shape &Line Color', self.chshapeLineColor,
                                icon='color_line', tip=u'Change the line color for this specific shape',
                                enabled=False)
        shapeFillColor = action('Shape &Fill Color', self.chshapeFillColor,
                                icon='color', tip=u'Change the fill color for this specific shape',
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText('Show/Hide Label Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Lavel list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Store actions for further handling.
        self.actions = struct(save=save, saveAs=saveAs, open=open, close=close,
                              lineColor=color1, fillColor=color2,
                              create=create, createRo=createRo, delete=delete, edit=edit, copy=copy,
                              createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              openNextLabeledImg=openNextLabeledImg,
                              openPrevLabeledImg=openPrevLabeledImg,
                              openNextChangedLabeledImg=openNextChangedLabeledImg,
                              openPrevChangedLabeledImg=openPrevChangedLabeledImg,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, setOutputDirsAction, copyToOutputAction, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, color2),
                              beginnerContext=(create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        addActions(self.menus.file,
                   (open, opendir, changeSavedir, openAnnotation, self.menus.recentFiles, save, saveAs, close, setOutputDirsAction, copyToOutputAction, None, quit))
        addActions(self.menus.help, (help,))
        addActions(self.menus.view, (
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))
        # 将新的动作添加到 File 菜单
        addActions(self.menus.file, (openNextLabeledImg, openPrevLabeledImg))
        addActions(self.menus.file, (openNextChangedLabeledImg, openPrevChangedLabeledImg))

        # Add these actions to a menu
        addActions(self.menus.view, (setSmallAnnotationThresholdsAction, jumpToSmallAnnotationsAction, jumpToSmallAnnotationsBackwardAction))

        # Add this where other toolbar actions are added
        smallAnnotationsToolBar = self.toolbar('Small Annotations')
        addActions(smallAnnotationsToolBar, (setSmallAnnotationThresholdsAction, jumpToSmallAnnotationsAction, jumpToSmallAnnotationsBackwardAction))

        # Add to the shortcuts list
        shortcuts.extend([
            (self.jumpToSmallAnnotations, "Ctrl+J"),
            (lambda: self.jumpToSmallAnnotations(backward=True), "Ctrl+Shift+J")
        ])

        self.menus.file.aboutToShow.connect(self.updateFileMenu)


        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')

        # Add these actions to the file menu
        addActions(self.menus.file, (jumpToDamAction, jumpToDamBackwardAction, jumpToUnlabeledAction, jumpToUnlabeledBackwardAction))

        # Or if you prefer, add them to the existing toolBar
        # Create a separate jump navigation section in the toolbar
        self.jumpToolBar = self.toolbar('Jump Navigation')
        addActions(self.jumpToolBar, (jumpToDamAction, jumpToDamBackwardAction, jumpToUnlabeledAction, jumpToUnlabeledBackwardAction))


        # Add this action to the view menu
        addActions(self.menus.view, (zoomToShapeAction,))

        # And optionally add it to the toolbar
        addActions(self.tools, (zoomToShapeAction,))

        self.actions.beginner = (
            open, opendir, openNextImg, openPrevImg, next_three_img, back_three_img, verify, save, None, create, createRo, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            open, save, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.image_1 = QImage()
        self.image_2 = QImage()
        self.filePath = ustr(defaultFilename)
        self.filePath_1 = ustr(defaultFilename)
        self.filePath_2 = ustr(defaultFilename)
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)
        # XXX: Could be completely declarative.
        # Restore application settings.
        if have_qstring():
            types = {
                'filename': QString,
                'recentFiles': QStringList,
                'window/size': QSize,
                'window/position': QPoint,
                'window/geometry': QByteArray,
                'line/color': QColor,
                'fill/color': QColor,
                'advanced': bool,
                # Docks and toolbars:
                'window/state': QByteArray,
                'savedir': QString,
                'lastOpenDir': QString,
            }
        else:
            types = {
                'filename': str,
                'recentFiles': list,
                'window/size': QSize,
                'window/position': QPoint,
                'window/geometry': QByteArray,
                'line/color': QColor,
                'fill/color': QColor,
                'advanced': bool,
                # Docks and toolbars:
                'window/state': QByteArray,
                'savedir': str,
                'lastOpenDir': str,
            }

        self.settings = settings = Settings(types)
        self.recentFiles = list(settings.get('recentFiles', []))
        size = settings.get('window/size', QSize(600, 500))
        position = settings.get('window/position', QPoint(0, 0))
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get('savedir', None))
        self.lastOpenDir = ustr(settings.get('lastOpenDir', None))
        if saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        # or simply:
        # self.restoreGeometry(settings['window/geometry']
        self.restoreState(settings.get('window/state', QByteArray()))
        self.lineColor = QColor(settings.get('line/color', Shape.line_color))
        self.fillColor = QColor(settings.get('fill/color', Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get('advanced', False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()
        # Since loading the file may take some time, make sure it runs in the
        # background.
        self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()
        self.printScrollPosition(self.scroll, self.canvas)
        self.printScrollPosition(self.scroll_1, self.canvas_1)
        self.printScrollPosition(self.scroll_2, self.canvas_2)
    
    def setOutputDirs(self, _value=False):
        """Set the output directories for images and labels."""
        # Ask for image output directory
        imgDirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                    '%s - Choose Output Image Directory' % __appname__, 
                                                    self.lastOpenDir or '.',  
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        
        if not imgDirpath or len(imgDirpath) <= 1:
            return
        
        # Ask for label output directory
        labelDirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                    '%s - Choose Output Label Directory' % __appname__, 
                                                    self.lastOpenDir or '.',  
                                                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        
        if not labelDirpath or len(labelDirpath) <= 1:
            return
        
        # Set the output directories
        self.outputImageDir = imgDirpath
        self.outputLabelDir = labelDirpath
        
        self.statusBar().showMessage('Output directories set: Images to %s, Labels to %s' % 
                                    (self.outputImageDir, self.outputLabelDir))


    def copyToOutputDirs(self):
        """Copy current image group and corresponding XML files to output directories."""
        # Check if output directories are set
        if not self.outputImageDir or not self.outputLabelDir:
            self.statusBar().showMessage('Output directories not set. Please set them first.')
            return
        
        # Check if we have a current file path
        if not self.filePath:
            self.statusBar().showMessage('No image group loaded.')
            return
        
        # Find the current group
        current_group = None
        for prefix, image_paths in self.mImgList:
            if self.filePath in image_paths:
                current_group = (prefix, image_paths)
                break
        
        if not current_group:
            self.statusBar().showMessage('Error: Could not find current image group.')
            return
        
        # Copy images to output image directory
        prefix, image_paths = current_group
        copied_images = 0
        copied_xmls = 0
        
        for img_path in image_paths:
            # Copy image file
            img_filename = os.path.basename(img_path)
            dest_img_path = os.path.join(self.outputImageDir, img_filename)
            try:
                shutil.copy2(img_path, dest_img_path)
                copied_images += 1
            except Exception as e:
                self.statusBar().showMessage(f'Error copying image {img_filename}: {str(e)}')
                continue
            
            # Copy XML file
            # xml_filename = os.path.splitext(img_filename)[0] + XML_EXT
            xml_filename = os.path.splitext(img_filename)[0] + ".jpg.aux.xml"
            image_folder = os.path.dirname(img_path)
            src_xml_path = os.path.join(image_folder, xml_filename) if image_folder else os.path.splitext(img_path)[0] + ".jpg.aux.xml"
            dest_xml_path = os.path.join(self.outputImageDir, xml_filename)
            print(f"src_xml_path: {src_xml_path}, dest_xml_path: {dest_xml_path}")

            if os.path.exists(src_xml_path):
                try:
                    shutil.copy2(src_xml_path, dest_xml_path)
                    copied_xmls += 1
                except Exception as e:
                    self.statusBar().showMessage(f'Error copying XML {xml_filename}: {str(e)}')

            # Copy txt file
            txt_filename = os.path.splitext(img_filename)[0] + '.txt'
            src_txt_path = os.path.join(self.defaultSaveDir, txt_filename) if self.defaultSaveDir else os.path.splitext(img_path)[0] + '.txt'
            dest_txt_path = os.path.join(self.outputLabelDir, txt_filename)
            print(f"src_txt_path: {src_txt_path}, dest_txt_path: {dest_txt_path}")

            if os.path.exists(src_txt_path):
                try:
                    shutil.copy2(src_txt_path, dest_txt_path)
                    copied_xmls += 1
                except Exception as e:
                    self.statusBar().showMessage(f'Error copying XML {txt_filename}: {str(e)}')
        
        # Show success message
        self.statusBar().showMessage(f'Copied {copied_images} images and {copied_xmls} XML files to {self.outputImageDir} , Copied txt to {self.outputLabelDir}')
        
    def checkSmallAnnotations(self, threshold_width=30, threshold_height=30):
        """
        Check for annotations that are smaller than the threshold dimensions.
        
        Args:
            threshold_width: Minimum width in pixels
            threshold_height: Minimum height in pixels
        
        Returns:
            True if small annotations found, False otherwise
        """
        # Check across all three canvases
        for canvas in [self.canvas, self.canvas_1, self.canvas_2]:
            for shape in canvas.shapes:
                # Get the bounding rect dimensions
                rect = shape.boundingRect()
                width = rect.width()
                height = rect.height()
                
                # Check if either dimension is below threshold
                if width < threshold_width or height < threshold_height:
                    return True
        
        return False

    def setSmallAnnotationThresholds(self):
        """
        Open a dialog to set thresholds for small annotation detection.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Small Annotation Thresholds")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout()
        
        # Width threshold
        width_layout = QHBoxLayout()
        width_label = QLabel("Minimum Width (pixels):")
        width_spinbox = QSpinBox()
        width_spinbox.setRange(1, 1000)
        width_spinbox.setValue(self.small_annotation_width_threshold 
                            if hasattr(self, "small_annotation_width_threshold") else 30)
        width_layout.addWidget(width_label)
        width_layout.addWidget(width_spinbox)
        
        # Height threshold
        height_layout = QHBoxLayout()
        height_label = QLabel("Minimum Height (pixels):")
        height_spinbox = QSpinBox()
        height_spinbox.setRange(1, 1000)
        height_spinbox.setValue(self.small_annotation_height_threshold 
                                if hasattr(self, "small_annotation_height_threshold") else 30)
        height_layout.addWidget(height_label)
        height_layout.addWidget(height_spinbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        # Add all layouts to main layout
        layout.addLayout(width_layout)
        layout.addLayout(height_layout)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Connect signals
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            self.small_annotation_width_threshold = width_spinbox.value()
            self.small_annotation_height_threshold = height_spinbox.value()
            self.statusBar().showMessage(f"Small annotation thresholds set to {self.small_annotation_width_threshold}x{self.small_annotation_height_threshold} pixels")
            return True
        
        return False


    def jumpToSmallAnnotations_ori(self, backward=False, threshold_width=30, threshold_height=30):
        """
        Jump to next/previous image group with annotations smaller than threshold.
        
        Args:
            backward: If True, search in reverse direction
            threshold_width: Minimum width in pixels
            threshold_height: Minimum height in pixels
        """
        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # Find the current group index
        current_group_index = -1
        if self.filePath:
            for i, (prefix, image_paths) in enumerate(self.mImgList):
                if self.filePath in image_paths:
                    current_group_index = i
                    break

        # If we couldn't find the current index, start at the beginning or end
        if current_group_index == -1:
            current_group_index = len(self.mImgList) - 1 if backward else 0
        
        # Determine the starting index and step direction
        if backward:
            start_index = (current_group_index - 1) % len(self.mImgList)
            step = -1
        else:
            start_index = (current_group_index + 1) % len(self.mImgList)
            step = 1

        index = start_index
        
        # Track our progress to avoid infinite loop
        checked_indices = set()
        
        while index not in checked_indices:
            checked_indices.add(index)
            
            # Load the image group to check annotations
            self.loadFile(groupIndex=index)
            
            # Check if any annotations are smaller than threshold
            if self.checkSmallAnnotations(threshold_width, threshold_height):
                self.statusBar().showMessage(f"Found image group with small annotations (below {threshold_width}x{threshold_height} pixels)")
                return
            
            # Move to the next/previous index
            index = (index + step) % len(self.mImgList)
            
            # If we've checked all groups or returned to start, break the loop
            if index == start_index or len(checked_indices) >= len(self.mImgList):
                direction = "previous" if backward else "next"
                self.statusBar().showMessage(f"No {direction} image groups with small annotations found")
                break

    def jumpToSmallAnnotations(self, backward=False, threshold_width=None, threshold_height=None):
        """
        Jump to the next/previous image group with small annotations.
        
        Args:
            backward: If True, search in reverse direction
            threshold_width: Minimum width in pixels (uses saved value if None)
            threshold_height: Minimum height in pixels (uses saved value if None)
        """
        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return
        
        # Use saved thresholds if not specified
        if threshold_width is None:
            threshold_width = self.small_annotation_width_threshold
        if threshold_height is None:
            threshold_height = self.small_annotation_height_threshold
        
        # Find the current group index
        current_group_index = -1
        if self.filePath:
            for i, (prefix, image_paths) in enumerate(self.mImgList):
                if self.filePath in image_paths:
                    current_group_index = i
                    break
        
        # If we couldn't find the current index, start at the beginning or end
        if current_group_index == -1:
            current_group_index = len(self.mImgList) - 1 if backward else 0
        
        # Determine the starting index and step direction
        if backward:
            start_index = (current_group_index - 1) % len(self.mImgList)
            step = -1
        else:
            start_index = (current_group_index + 1) % len(self.mImgList)
            step = 1

        # Create a progress dialog
        progress = QProgressDialog(
            f"Searching for annotations smaller than {threshold_width}x{threshold_height} pixels...", 
            "Cancel", 0, len(self.mImgList), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(500)  # Only show for operations taking > 500ms
        
        # Start searching from the next/previous group
        index = start_index
        groups_checked = 0
        
        try:
            while groups_checked < len(self.mImgList):
                # Update progress
                progress.setValue(groups_checked)
                if progress.wasCanceled():
                    break
                
                # Check for small annotations in this group
                prefix, image_paths = self.mImgList[index]
                has_small_annotation = False
                small_annotation_count = 0
                
                for img_path in image_paths:
                    annotation_filename = os.path.splitext(os.path.basename(img_path))[0] + XML_EXT
                    annotation_path = os.path.join(self.defaultSaveDir, annotation_filename) if self.defaultSaveDir else os.path.splitext(img_path)[0] + XML_EXT
                    
                    if os.path.exists(annotation_path):
                        # Parse the XML to check annotation dimensions
                        try:
                            reader = DotaReader(annotation_path)
                            shapes = reader.getShapes()
                            
                            for _, points, _, _, _, _, _ in shapes:
                                # Calculate width and height from points
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                width = max(x_coords) - min(x_coords)
                                height = max(y_coords) - min(y_coords)
                                
                                if width < threshold_width or height < threshold_height:
                                    has_small_annotation = True
                                    small_annotation_count += 1
                        except Exception as e:
                            self.statusBar().showMessage(f"Error parsing annotation file: {str(e)}")
                            continue
                
                # If we found a group with small annotations, jump to it
                if has_small_annotation:
                    # Close progress dialog before loading new file
                    progress.close()
                    
                    # Important: Store the target index
                    target_index = index
                    
                    # Use a separate method call with a timer to ensure UI updates properly
                    QTimer.singleShot(100, lambda: self._jumpToTargetGroup(target_index, small_annotation_count, threshold_width, threshold_height))
                    return True
                
                # Move to the next/previous index
                index = (index + step) % len(self.mImgList)
                groups_checked += 1
        
        finally:
            progress.close()
        
        # If we checked all groups and didn't find any small annotations
        direction = "previous" if backward else "next"
        self.statusBar().showMessage(f"No {direction} image groups with annotations smaller than {threshold_width}x{threshold_height} pixels found")
        return False

    def _jumpToTargetGroup(self, groupIndex, small_annotation_count, threshold_width, threshold_height):
        """
        Helper method to perform the actual jump to the target group and handle post-jump actions.
        
        Args:
            groupIndex: Index of the target group to jump to
            small_annotation_count: Number of small annotations found
            threshold_width: Width threshold used
            threshold_height: Height threshold used
        """
        # Get group info for messaging
        prefix, _ = self.mImgList[groupIndex]
        
        # Load the target group
        success = self.loadFile(groupIndex=groupIndex)
        
        if success:
            # After loading, find and select a small annotation
            for canvas in [self.canvas, self.canvas_1, self.canvas_2]:
                for shape in canvas.shapes:
                    rect = shape.boundingRect()
                    width = rect.width()
                    height = rect.height()
                    
                    if width < threshold_width or height < threshold_height:
                        # Select the shape
                        canvas.selectShape(shape)
                        canvas.update()
                        
                        # If auto zoom is enabled, zoom to this shape
                        if hasattr(self, 'autoZoomEnabled') and self.autoZoomEnabled:
                            self.zoomToShape(shape)
                        
                        # Show success message
                        self.statusBar().showMessage(
                            f"Found group '{prefix}' with {small_annotation_count} annotation(s) smaller than {threshold_width}x{threshold_height} pixels"
                        )
                        return
            
            # If we didn't find/select a small annotation, still show success message
            self.statusBar().showMessage(
                f"Found group '{prefix}' with {small_annotation_count} annotation(s) smaller than {threshold_width}x{threshold_height} pixels"
            )
        else:
            self.statusBar().showMessage(f"Error: Failed to load image group at index {groupIndex}")

    def getRandomEncouragementMessage(self, percentage):
        """
        Get a random encouraging message based on progress percentage, enhanced with emojis.
        """
        import random
        
        starting_messages = [
            "🌟 Great start! Every annotation counts!",
            "🚀 You're making a difference with each label!",
            "💪 Building momentum! Keep going!",
            "🐾 You've got this! One label at a time!"
        ]
        
        middle_messages = [
            "📈 You're making steady progress! Keep it up!",
            "🌈 Look how far you've come already!",
            "🔥 Your consistency is impressive!",
            "💖 You're doing amazing work!"
        ]
        
        finishing_messages = [
            "🏁 The finish line is getting closer!",
            "🌼 Your dedication is remarkable!",
            "🎯 Almost there! You can do it!",
            "⚡ You're crushing these annotations!"
        ]
        
        complete_messages = [
            "🎉 Fantastic achievement! You did it!",
            "🏆 Outstanding work completing all annotations!",
            "💎 Your contribution to this project is invaluable!",
            "🥳 Success! Celebrate this accomplishment!"
        ]
        
        if percentage < 25:
            return random.choice(starting_messages)
        elif percentage < 60:
            return random.choice(middle_messages)
        elif percentage < 100:
            return random.choice(finishing_messages)
        else:
            return random.choice(complete_messages)


    def updateProgressDisplay(self):
        """
        Update the progress display with motivational messages and emojis.
        """
        if self.totalImageGroups == 0:
            # Count total image groups if not done yet
            self.totalImageGroups = len(self.mImgList)
            
        # Count completed image groups
        self.countCompletedGroups()
        
        # Calculate percentage
        if self.totalImageGroups > 0:
            percentage = (self.completedImageGroups / self.totalImageGroups) * 100
        else:
            percentage = 0
        
        # Get encouraging message
        encouragement = self.getRandomEncouragementMessage(percentage)
        
        # Format the progress message
        message = f"🌟 Great job! You've completed {percentage:.1f}% of all image groups!\n{encouragement}"
        
        # Display progress and message
        self.progressLabel.setText(message)
        
        # Add progress info to status bar as well
        self.statusBar().showMessage(f"📊 Progress: {self.completedImageGroups}/{self.totalImageGroups} groups labeled ({percentage:.1f}%)", 5000)


    def countCompletedGroups(self):
        """
        Calculate progress based on the current position in the file list.
        """
        # If file list is empty, return 0
        if len(self.mImgList) == 0:
            self.completedImageGroups = 0
            return
        
        # Find the current position in the file list
        current_position = 0
        if self.filePath:
            for i, (prefix, image_paths) in enumerate(self.mImgList):
                if self.filePath in image_paths:
                    current_position = i + 1  # Add 1 because we count from 1, not 0
                    break
        
        # Update completed count
        self.completedImageGroups = current_position
        self.totalImageGroups = len(self.mImgList)

    ## Support Functions ##
    def printScrollPosition(self, scroll, canvas):
        # 获取滚动条的几何信息
        geo = scroll.geometry()
        geo_c = canvas.geometry()

        # 获取滚动条在窗口中的位置（相对于窗口左上角的坐标）
        x = geo.x()
        y = geo.y()
        x_c = geo_c.x()
        y_c = geo_c.y()
        width = geo.width()
        height = geo.height()
        width_c = geo_c.width()
        height_c = geo_c.height()
        # print(scroll,f" bar position: x={x}, y={y}, width={width}, height={height}")
        # print(canvas, f" bar position: x={x_c}, y={y_c}, width={width_c}, height={height_c}")

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        print("toggleAdvancedMode")
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            # self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            pass
            # self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        print("populateModeActions")
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.canvas.verified = False
        self.canvas_1.verified = False
        self.canvas_2.verified = False
        self.actions.save.setEnabled(True)
        # print("setdirty")

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)
        self.actions.createRo.setEnabled(True)

    def enableCreate(self,b):
        print("enablecreate")
        self.isEnableCreate = not b
        self.actions.create.setEnabled(self.isEnableCreate)

    def enableCreateRo(self,b):
        print("enablecreateRo")
        self.isEnableCreateRo = not b
        self.actions.createRo.setEnabled(self.isEnableCreateRo)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        # print(message)
        # print("status")
        self.statusBar().showMessage(message, delay)
        self.statusBar().show()

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.imageData_1 = None
        self.imageData_2 = None
        self.labelFile = None
        self.canvas.resetState()
        # print("resetState")

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    ## Callbacks ##
    def tutorial(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    # create Normal Rect
    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.canDrawRotatedRect = False
        self.actions.create.setEnabled(False)
        self.actions.createRo.setEnabled(False)

    # create Rotated Rect
    def createRoShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.canvas.canDrawRotatedRect = True
        self.canvas_1.setEditing(False)
        self.canvas_1.canDrawRotatedRect = True
        self.canvas_2.setEditing(False)
        self.canvas_2.canDrawRotatedRect = True
        self.actions.create.setEnabled(False)
        self.actions.createRo.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        print("toggleDrawingSensitive")
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.canvas_1.setEditing(True)
            self.canvas_1.restoreCursor()
            self.canvas_2.setEditing(True)
            self.canvas_2.restoreCursor()
            self.actions.create.setEnabled(True)
            self.actions.createRo.setEnabled(True)
            

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        print('setCreateMode')
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self, item=None):
        print("editlabel")
        if not self.canvas.editing():
            return
        item = item if item else self.currentItem()
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            self.setDirty()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked_ori(self, item=None):
        if not self.mayContinue():
            return

        if not item:
            return

        groupIndex = self.fileListWidget.row(item)
        if groupIndex >= 0 and groupIndex < len(self.mImgList):
            self.loadFile(groupIndex)

    def updateAnnotationCount(self):
        """
        Update the displayed annotation counts for all three canvases,
        including the total count across all canvases.
        """
        # Count annotations in each canvas
        count_2010 = len(self.canvas.shapes)
        count_2015 = len(self.canvas_1.shapes)
        count_2020 = len(self.canvas_2.shapes)
        
        # Calculate the total count
        total_count = count_2010 + count_2015 + count_2020
        
        # Format the counts as a string including the total
        count_text = f"Annotations: 2010: {count_2010} | 2015: {count_2015} | 2020: {count_2020} | Total: {total_count}"
        
        # Update the status bar with the counts
        self.statusBar().showMessage(count_text, 5000)  # Show for 5 seconds
        
        # If you prefer a permanent display instead of status bar, update the label
        if hasattr(self, 'annotationCountLabel'):
            self.annotationCountLabel.setText(count_text)

    def extract_coordinates_from_xml(self, xml_path):
        """
        Parse the XML file and extract the coordinates of the upper-right corner.
        
        Args:
            xml_path (str): Path to the XML file
            
        Returns:
            tuple: (longitude, latitude) of the upper-right corner or None if parsing fails
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract the GeoTransform element
            geo_transform_elem = root.find('GeoTransform')
            if geo_transform_elem is None:
                return None
                
            # Parse the GeoTransform values
            geo_transform = [float(x) for x in geo_transform_elem.text.strip().split(',')]
            
            # GeoTransform format: [x0, dx, 0, y0, 0, dy]
            x0 = geo_transform[0]  # Upper-left x (longitude)
            dx = geo_transform[1]  # W-E pixel resolution
            y0 = geo_transform[3]  # Upper-left y (latitude)
            dy = geo_transform[5]  # N-S pixel resolution (negative value)
            
            # To calculate the upper-right corner coordinates:
            # For an image with width 'w' and height 'h'
            # Upper-right longitude = x0 + w * dx
            # Upper-right latitude = y0
            
            # Since we don't have image dimensions in the XML, we need to get them from the image
            # For now, let's assume we have the image width
            # We'll get it from the actual image when integrating this into the main code
            
            return (x0, y0)
            
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return None
    

    def updateCoordinatesDisplay(self):
        """
        Update the coordinates display with coordinates from the XML file
        corresponding to the current image.
        """
        if self.filePath is None:
            self.coordsLabel.setText("")
            return
            
        # Build the path to the XML file based on the image path
        xml_path = os.path.splitext(self.filePath)[0] + ".jpg.aux.xml"
        print(f"XML path: {xml_path}")
        
        if not os.path.exists(xml_path):
            self.coordsLabel.setText("No coordinates data available")
            return
            
        # Extract image dimensions
        image_width = self.image.width()
        image_height = self.image.height()
        
        # Extract coordinates from XML
        base_coords = self.extract_coordinates_from_xml(xml_path)
        if base_coords is None:
            self.coordsLabel.setText("Failed to parse coordinates")
            return
            
        x0, y0 = base_coords
        
        # We need to find the GeoTransform values to calculate upper-right coordinates
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            geo_transform_elem = root.find('GeoTransform')
            if geo_transform_elem is not None:
                geo_transform = [float(x.strip()) for x in geo_transform_elem.text.split(',')]
                dx = geo_transform[1]  # W-E pixel resolution
                dy = geo_transform[5]  # N-S pixel resolution (negative value)
                
                # Calculate upper-right coordinates
                lon_ur = x0 + (image_width * dx)
                lat_ur = y0 + (image_height * dy)  # dy is negative, so this is effectively subtraction
                
                # Format coordinates to display with appropriate precision
                coords_text = f"{lat_ur:.6f}, {lon_ur:.6f}"
                self.coordsLabel.setText(coords_text)
            else:
                self.coordsLabel.setText("No GeoTransform data found")
        except Exception as e:
            self.coordsLabel.setText(f"Error processing coordinates: {str(e)}")


    def copyCoordinatesToClipboard(self):
        """
        Copy the displayed coordinates to the clipboard
        """
        text = self.coordsLabel.text()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.statusBar().showMessage("Coordinates copied to clipboard", 2000)

    def fileitemDoubleClicked(self, item=None):
        if not self.mayContinue():
            return

        if not item:
            return

        # 查找文件列表中的组，比较显示名称（前缀）是否匹配
        groupIndex = None
        # print(self.mImgList)
        # print(ustr(item.text()))
        for idx, (prefix, image_paths) in enumerate(self.mImgList):
            display_name = prefix.title()  # 格式化显示名称
            # print(display_name)
            if display_name == ustr(item.text()):
                groupIndex = idx
                break

        if groupIndex is not None:
            self.loadFile(groupIndex)
        else:
            print(f"未找到对应的组: {ustr(item.text())}")

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
         date on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        print("shapeSelectionChanged")
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)


    def selection_changed(self):
        if self.canvas_1.selectedShape or self.canvas_2.selectedShape:
            self.ignore_label_selection = True
            self.labelList.clearSelection()
            self.ignore_label_selection = False
            self.on_canvas_selection_changed()

    def on_canvas_selection_changed(self, selected=False):
        # print("on_canvas_selection_changed")
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            self.ignore_label_selection = True  # 在整个选择更改过程中保持

            if shape:
                # print("Canvas selection changed to shape:", shape.label)
                self.canvas_1.selectedShape = None
                self.canvas_2.selectedShape = None
                self.update_canvas_selection()  # 更新 canvas 的选择状态
                self.display_year_labels()
                # 设置标志位，避免触发 labelSelectionChanged
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                if shape in self.shapesToItems:
                    self.shapesToItems[shape].setSelected(True)
                self.ignore_label_selection = False
                item = self.shapesToItems.get(shape)
                if item:
                    # print(f"Selected item: {item.text()}")  # 调试信息
                    # 增加延迟时间
                    # QTimer.singleShot(10, lambda: self.scroll_to_item(item))
                    persistent_index = QPersistentModelIndex(self.labelList.indexFromItem(item))
                    QTimer.singleShot(10, lambda: self.scroll_to_persistent_index(persistent_index))

            else:
                # print("Canvas selection cleared")
                # 清除选择时，也设置标志位
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                self.ignore_label_selection = False

        # 更新动作的可用状态
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def on_canvas_selection_changed_1(self, selected=False):
        # print("on_canvas_selection_changed")
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:

            shape_1 = self.canvas_1.selectedShape

            self.ignore_label_selection = True  # 在整个选择更改过程中保持


            if shape_1:
                # print("Canvas_1 selection changed to shape:", shape_1.label)
                self.canvas.selectedShape = None
                self.canvas_2.selectedShape = None
                self.update_canvas_selection()  # 更新 canvas 的选择状态

                self.display_year_labels()
                # 设置标志位，避免触发 labelSelectionChanged
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                if shape_1 in self.shapesToItems:
                    self.shapesToItems[shape_1].setSelected(True)
                self.ignore_label_selection = False
                item = self.shapesToItems.get(shape_1)
                if item:
                    # print(f"Selected item: {item.text()}")  # 调试信息
                    # 增加延迟时间
                    # QTimer.singleShot(10, lambda: self.scroll_to_item(item))
                    persistent_index = QPersistentModelIndex(self.labelList.indexFromItem(item))
                    QTimer.singleShot(10, lambda: self.scroll_to_persistent_index(persistent_index))

            else:
                # print("Canvas selection cleared")
                # 清除选择时，也设置标志位
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                self.ignore_label_selection = False

        # 更新动作的可用状态
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def on_canvas_selection_changed_2(self, selected=False):
        # print("on_canvas_selection_changed")
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:

            shape_2 = self.canvas_2.selectedShape

            self.ignore_label_selection = True  # 在整个选择更改过程中保持

            if shape_2:
                # print("Canvas_1 selection changed to shape:", shape_2.label)
                self.canvas_1.selectedShape = None
                self.canvas.selectedShape = None
                self.update_canvas_selection()  # 更新 canvas 的选择状态

                self.display_year_labels()
                # 设置标志位，避免触发 labelSelectionChanged
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                if shape_2 in self.shapesToItems:
                    self.shapesToItems[shape_2].setSelected(True)
                self.ignore_label_selection = False
                item = self.shapesToItems.get(shape_2)
                if item:
                    # print(f"Selected item: {item.text()}")  # 调试信息
                    # 增加延迟时间
                    # QTimer.singleShot(10, lambda: self.scroll_to_item(item))
                    persistent_index = QPersistentModelIndex(self.labelList.indexFromItem(item))
                    QTimer.singleShot(10, lambda: self.scroll_to_persistent_index(persistent_index))

            else:
                # print("Canvas selection cleared")
                # 清除选择时，也设置标志位
                self.ignore_label_selection = True
                self.labelList.clearSelection()
                self.ignore_label_selection = False

        # 更新动作的可用状态
        self.actions.delete.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def update_canvas_selection(self):
        """
        更新各个 Canvas 选择状态的显示
        """
        # 确保每个 canvas 上的选中状态能及时更新
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()

    def addLabel(self, shape):
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]

    def loadLabels_old(self, shapes, shapes_1, shapes_2):
        s = []
        s_1 = []
        s_2 = []
        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s.append(shape)
            self.addLabel(shape)
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes_1:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s_1.append(shape)
            self.addLabel(shape)
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes_2:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s_2.append(shape)
            self.addLabel(shape)

            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        self.canvas.loadShapes(s)
        self.canvas_1.loadShapes(s_1)
        self.canvas_2.loadShapes(s_2)


    def loadLabels(self, shapes, shapes_1, shapes_2):
        s = []
        s_1 = []
        s_2 = []
        # 为每个画布创建独立的字典分类标签
        category_dict_1 = defaultdict(list)  # 第一个画布
        category_dict_2 = defaultdict(list)  # 第二个画布
        category_dict_3 = defaultdict(list)  # 第三个画布
        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s.append(shape)
            category_dict_1[label].append(shape)  # 第一个画布按类别分类
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes_1:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s_1.append(shape)
            category_dict_2[label].append(shape)  # 第二个画布按类别分类
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        for label, points, direction, isRotated, line_color, fill_color, difficult in shapes_2:
            shape = Shape(label=label)
            for x, y in points:
                shape.addPoint(QPointF(x, y))
            shape.difficult = difficult
            shape.direction = direction
            shape.isRotated = isRotated
            shape.close()
            s_2.append(shape)
            category_dict_3[label].append(shape)  # 第三个画布按类别分类
            if line_color:
                shape.line_color = QColor(*line_color)
            if fill_color:
                shape.fill_color = QColor(*fill_color)

        # 按类别排序并将标签添加到 labelList 中
        sorted_categories = sorted(set(category_dict_1.keys()).union(category_dict_2.keys(), category_dict_3.keys()))  # 合并所有画布的类别
        for category in sorted_categories:
            for shape in category_dict_1[category] + category_dict_2[category] + category_dict_3[category]:
                item = HashableQListWidgetItem(shape.label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.itemsToShapes[item] = shape
                self.shapesToItems[shape] = item
                self.labelList.addItem(item)

        # 加载标签到相应的画布
        self.canvas.loadShapes(s)  # 加载第一个画布的形状
        self.canvas_1.loadShapes(s_1)  # 加载第二个画布的形状
        self.canvas_2.loadShapes(s_2)  # 加载第三个画布的形状

        self.updateAnnotationCount()

    def saveLabels(self, annotationFilePath, canvas):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = canvas.verified

        def format_shape(s):
            return dict(label=s.label,
                        line_color=s.line_color.getRgb()
                        if s.line_color != self.lineColor else None,
                        fill_color=s.fill_color.getRgb()
                        if s.fill_color != self.fillColor else None,
                        points=[(p.x(), p.y()) for p in s.points],
                       # add chris
                        difficult = s.difficult,
                        # You Hao 2017/06/21
                        # add for rotated bounding box
                        direction = s.direction,
                        center = s.center,
                        isRotated = s.isRotated)

        shapes = [format_shape(shape) for shape in canvas.shapes]
        # Can add differrent annotation formats here
        try:
            print ('Img: ' + self.filePath + ' -> Its txt: ' + annotationFilePath)
            with open(annotationFilePath, 'w') as f:
                for shape in shapes:
                    points = shape['points']
                    label = shape['label']
                    difficult = 0
                    # 将 4 个点坐标 + 标签 + 难度级别写入文件
                    line = " ".join([f"{p[0]} {p[1]}" for p in points]) + f" {label} {difficult}\n"
                    f.write(line)
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data',
                              u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        """
        Copy the selected shape from any of the three canvases
        """
        # Try to copy from whichever canvas has a selected shape
        if self.canvas.selectedShape:
            shape = self.canvas.copySelectedShape()
        elif self.canvas_1.selectedShape:
            shape = self.canvas_1.copySelectedShape()
        elif self.canvas_2.selectedShape:
            shape = self.canvas_2.copySelectedShape()
        else:
            return  # No shape selected in any canvas

        if shape:  # If a shape was copied
            self.addLabel(shape)
            # Fix copy and delete
            self.shapeSelectionChanged(True)

    def labelSelectionChanged_old(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)

    def jumpToDamLabel(self, backward=False):
        """
        Jump to image groups that have gravity, arch, or barrage dam labels.
        
        Args:
            backward (bool): If True, search backwards from current position
        """
        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return
        
        # Find the current group index
        current_group_index = -1
        if self.filePath:
            for i, (prefix, image_paths) in enumerate(self.mImgList):
                if self.filePath in image_paths:
                    current_group_index = i
                    break

        # If we couldn't find the current index, start at the beginning or end
        if current_group_index == -1:
            current_group_index = len(self.mImgList) - 1 if backward else 0
        
        # Determine the starting index and step direction
        if backward:
            start_index = (current_group_index - 1) % len(self.mImgList)
            step = -1
        else:
            start_index = (current_group_index + 1) % len(self.mImgList)
            step = 1

        index = start_index
        # Keywords to search for in the labels
        target_labels = ["gravity", "arch", "barrage"]

        while True:
            # Get the group at the current index
            prefix, image_paths = self.mImgList[index]
            found = False

            # Check if any of the images in this group have the target labels
            for img_path in image_paths:
                annotation_filename = os.path.splitext(os.path.basename(img_path))[0] + XML_EXT
                annotation_path = os.path.join(self.defaultSaveDir, annotation_filename) if self.defaultSaveDir else os.path.splitext(img_path)[0] + XML_EXT
                
                if os.path.exists(annotation_path):
                    try:
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            # Check if any of the target labels are in the file
                            if any(label in content for label in target_labels):
                                found = True
                                break
                    except:
                        continue

            if found:
                # Load this group and exit the loop
                self.loadFile(groupIndex=index)
                direction = "previous" if backward else "next"
                self.statusBar().showMessage(f"Found {direction} dam label in group: {prefix}")
                return
            
            # Move to the next/previous index
            index = (index + step) % len(self.mImgList)
            
            # If we've checked all groups, break the loop
            if index == start_index:
                direction = "previous" if backward else "next"
                self.statusBar().showMessage(f"No {direction} image group with gravity, arch, or barrage dam labels found")
                break

    def jumpToDamLabelBackward(self):
        """
        Jump to the previous image group that has gravity, arch, or barrage dam labels.
        """
        self.jumpToDamLabel(backward=True)

    def jumpToUnlabeledGroup(self, backward=False):
        """
        Jump to image groups that don't have any labels in all three years.
        
        Args:
            backward (bool): If True, search backwards from current position
        """
        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # Find the current group index
        current_group_index = -1
        if self.filePath:
            for i, (prefix, image_paths) in enumerate(self.mImgList):
                if self.filePath in image_paths:
                    current_group_index = i
                    break

        # If we couldn't find the current index, start at the beginning or end
        if current_group_index == -1:
            current_group_index = len(self.mImgList) - 1 if backward else 0
        
        # Determine the starting index and step direction
        if backward:
            start_index = (current_group_index - 1) % len(self.mImgList)
            step = -1
        else:
            start_index = (current_group_index + 1) % len(self.mImgList)
            step = 1

        index = start_index

        while True:
            # Get the group at the current index
            prefix, image_paths = self.mImgList[index]
            
            # Check if none of the images in this group have labels
            all_unlabeled = True
            for img_path in image_paths:
                annotation_filename = os.path.splitext(os.path.basename(img_path))[0] + XML_EXT
                annotation_path = os.path.join(self.defaultSaveDir, annotation_filename) if self.defaultSaveDir else os.path.splitext(img_path)[0] + XML_EXT
                
                if os.path.exists(annotation_path):
                    try:
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            if f.read().strip():  # If file exists and has content
                                all_unlabeled = False
                                break
                    except:
                        continue

            if all_unlabeled:
                # Load this group and exit the loop
                self.loadFile(groupIndex=index)
                direction = "previous" if backward else "next"
                self.statusBar().showMessage(f"Found {direction} unlabeled group: {prefix}")
                return
            
            # Move to the next/previous index
            index = (index + step) % len(self.mImgList)
            
            # If we've checked all groups, break the loop
            if index == start_index:
                direction = "previous" if backward else "next"
                self.statusBar().showMessage(f"No {direction} completely unlabeled image groups found")
                break

    def jumpToUnlabeledGroupBackward(self):
        """
        Jump to the previous image group that doesn't have any labels in all three years.
        """
        self.jumpToUnlabeledGroup(backward=True)
       


    def scroll_to_item_ori(self, item):
        self.labelList.scrollToItem(item, QAbstractItemView.PositionAtCenter)            
    

    def scroll_to_item(self, item):
        if item and self.labelList.indexFromItem(item).isValid():  # 确保 item 仍然存在
            self.labelList.scrollToItem(item, QAbstractItemView.PositionAtCenter)

    def scroll_to_persistent_index(self, persistent_index):
        if persistent_index.isValid():
            index = QModelIndex(persistent_index)  # 显式转换回 QModelIndex
            self.labelList.scrollTo(index, QAbstractItemView.PositionAtCenter)


    def toggleAutoZoom(self, state):
        """
        Toggle automatic zooming when an annotation is selected.
        """
        self.autoZoomEnabled = (state == Qt.Checked)
        self.statusBar().showMessage(f"Auto zoom {'enabled' if self.autoZoomEnabled else 'disabled'}", 2000)



    def labelSelectionChanged(self):
        if self.ignore_label_selection:
            print("Ignored labelSelectionChanged due to ignore_label_selection flag")
            return
        selected_items = self.labelList.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        # self.canvas.selectShape(self.itemsToShapes[item])
        # shape = self.itemsToShapes.get(item)
        canvas = None
        shape = None
        if item in self.itemsToShapes:
            shape = self.itemsToShapes[item]
            if shape in self.canvas.shapes:
                canvas = self.canvas
            elif shape in self.canvas_1.shapes:
                canvas = self.canvas_1
            elif shape in self.canvas_2.shapes:
                canvas = self.canvas_2
        # 如果找到了对应的 canvas
        if canvas:
            self.labelList.clearSelection()
            canvas.selectShape(self.itemsToShapes[item])
            print(f"Selected shape: {shape.center}")

            # Only zoom if auto-zoom is enabled
            if self.autoZoomEnabled and shape:
                self.zoomToShape(shape)
            # print(f"Fill color: {shape.fill_color}")
            # print(f"Label: {shape.label}")
            # self.zoomToShape(shape)
        # print(f"Fill color: {shape.fill_color}")
        # print(f"Label: {shape.label}")

        # if shape:
        #     self.zoomToShape(shape)
            
    def zoomToCurrentShape(self):
        """
        Zoom to the currently selected shape in any of the canvases.
        """
        # Check each canvas for a selected shape
        selected_shape = None
        
        if self.canvas.selectedShape:
            selected_shape = self.canvas.selectedShape
        elif self.canvas_1.selectedShape:
            selected_shape = self.canvas_1.selectedShape
        elif self.canvas_2.selectedShape:
            selected_shape = self.canvas_2.selectedShape
            
        if selected_shape:
            self.zoomToShape(selected_shape)
            self.statusBar().showMessage("Zoomed to selected annotation", 2000)
        else:
            self.statusBar().showMessage("No annotation selected to zoom to", 2000)

    def zoomToShape(self, shape):
        if not shape:
            return

        # 获取形状的中心坐标
        center_x = shape.center.x()
        center_y = shape.center.y()

        self.current_center = QPoint(int(center_x), int(center_y))

        # 设置期望的缩放级别，例如200%
        desired_zoom = 200
        # self.canvas.setZoom(desired_zoom)
        self.setZoom(desired_zoom)  # 正确的调用

        # 使用单次定时器延迟设置滚动条位置
        QTimer.singleShot(0, lambda: self.centerScrollOn(center_x, center_y))

    def centerScrollOn(self, x, y):
        # 获取当前缩放因子
        zoom_factor = self.zoom_level / 100.0


        # 计算缩放后的中心坐标
        scaled_x = x * zoom_factor
        scaled_y = y * zoom_factor


        # 获取滚动区域的视口大小
        viewport_width = self.scroll.viewport().width()
        viewport_height = self.scroll.viewport().height()

        # 计算新的滚动条位置，使中心点位于视图中央
        new_x = scaled_x - viewport_width / 2
        new_y = scaled_y - viewport_height / 2

        # 获取滚动条对象
        h_bar = self.scroll.horizontalScrollBar()
        v_bar = self.scroll.verticalScrollBar()
        h_bar_1 = self.scroll_1.horizontalScrollBar()
        v_bar_1 = self.scroll_1.verticalScrollBar()
        h_bar_2 = self.scroll_2.horizontalScrollBar()
        v_bar_2 = self.scroll_2.verticalScrollBar()
        # 限制滚动条的位置在有效范围内
        new_x = max(0, min(new_x, h_bar.maximum()))
        new_y = max(0, min(new_y, v_bar.maximum()))
        # 设置滚动条的位置
        h_bar.setValue(int(new_x))
        v_bar.setValue(int(new_y))
        h_bar_1.setValue(int(new_x))
        v_bar_1.setValue(int(new_y))
        h_bar_2.setValue(int(new_x))
        v_bar_2.setValue(int(new_y))


    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """

        if not self.useDefautLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            text = self.labelDialog.popUp(text=self.prevLabelText)
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            self.addLabel(self.canvas.setLastLabel(text))
            if self.beginner():  # Switch to edit mode.
             
                self.canvas.setEditing(True)
                self.canvas_1.setEditing(True)
                self.canvas_2.setEditing(True)
                self.actions.create.setEnabled(self.isEnableCreate)
                self.actions.createRo.setEnabled(self.isEnableCreateRo)
            else:
                self.actions.editMode.setEnabled(True)
               
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
                
        else:
            # self.canvas.undoLastLine()
           
            self.canvas.resetAllLines()

        self.updateAnnotationCount()

    def newShape_1(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        if not self.useDefautLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            text = self.labelDialog.popUp(text=self.prevLabelText)
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            self.addLabel(self.canvas_1.setLastLabel(text))
            if self.beginner():  # Switch to edit mode.
                self.canvas_1.setEditing(True)
                self.canvas.setEditing(True)
                self.canvas_2.setEditing(True)
                self.actions.create.setEnabled(self.isEnableCreate)
                self.actions.createRo.setEnabled(self.isEnableCreateRo)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas_1.resetAllLines()

        self.updateAnnotationCount()


    def newShape_2(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        
        if not self.useDefautLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            text = self.labelDialog.popUp(text=self.prevLabelText)

        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        if text is not None:
            self.prevLabelText = text
            self.addLabel(self.canvas_2.setLastLabel(text))
            if self.beginner():  # Switch to edit mode.
                self.canvas_2.setEditing(True)
                self.canvas.setEditing(True)
                self.canvas_1.setEditing(True)
                self.actions.create.setEnabled(self.isEnableCreate)
                self.actions.createRo.setEnabled(self.isEnableCreateRo)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas_2.resetAllLines()
        self.updateAnnotationCount()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))
        bar_1 = self.scrollBars_1[orientation]
        bar_1.setValue(int(bar_1.value() + bar_1.singleStep() * units))
        bar_2 = self.scrollBars_2[orientation]
        bar_2.setValue(int(bar_2.value() + bar_2.singleStep() * units))


    def scrollRequest_1(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars_1[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))
      

    def scrollRequest_2(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars_2[orientation]
        bar.setValue(int(bar.value() + bar.singleStep() * units))
     


    def setZoom_old(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def setZoom(self, value):
   
        # 阻止信号递归
        self.zoomWidget.blockSignals(True)
        self.zoomWidget.setValue(int(value))

        self.zoomWidget.blockSignals(False)
        
        # 更新 zoom_level
        self.zoom_level = value

        # 取消其他缩放相关的模式
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM

        # 更新画布的缩放因子
        self.canvas.setScale(self.zoom_level / 100.0)  # 假设 Canvas 有 setScale 方法
        self.canvas_1.setScale(self.zoom_level / 100.0)  # 假设 Canvas 有 setScale 方法
        self.canvas_2.setScale(self.zoom_level / 100.0)  # 假设 Canvas 有 setScale 方法
        # 重绘画布
        self.paintCanvas()

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    # def zoomRequest(self, delta):
    #     units = delta / (8 * 15)
    #     scale = 10
    #     self.addZoom(scale * units)

    def update_scroll(self):
        # 更新当前视图中心
        self.on_scroll_changed(self.scroll)
        # 同步其他两个滚动条
        self.sync_scrollbars(self.scroll, self.scroll_1, self.scroll_2)

    def update_scroll_1(self):
        self.on_scroll_changed(self.scroll_1)
        self.sync_scrollbars(self.scroll_1, self.scroll, self.scroll_2)

    def update_scroll_2(self):
        self.on_scroll_changed(self.scroll_2)
        self.sync_scrollbars(self.scroll_2, self.scroll_1, self.scroll)


    def on_scroll_changed(self, scroll):
        """
        当滚动条的位置变化时，更新 current_center 为新的视图中心点。
        """
        # 计算视图的中心点在图像坐标中的位置
        h_scroll = scroll.horizontalScrollBar().value()
        v_scroll = scroll.verticalScrollBar().value()
        viewport_width = scroll.viewport().width()
        viewport_height = scroll.viewport().height()

        zoom_factor = self.zoom_level / 100.0

        # 视口中心点在图像坐标中的位置
        center_x = (h_scroll + viewport_width / 2) / zoom_factor
        center_y = (v_scroll + viewport_height / 2) / zoom_factor

        self.current_center = QPoint(int(center_x), int(center_y))
  

    # 同步三个滚动条
    def sync_scrollbars(self, scroll, scroll_1, scroll_2):
        # 获取当前滚动条的水平和垂直位置
        h_scroll = scroll.horizontalScrollBar().value()
        v_scroll = scroll.verticalScrollBar().value()

        # 设置其他两个滚动条的位置
        scroll_1.horizontalScrollBar().setValue(h_scroll)
        scroll_1.verticalScrollBar().setValue(v_scroll)
        scroll_2.horizontalScrollBar().setValue(h_scroll)
        scroll_2.verticalScrollBar().setValue(v_scroll)

 

    def zoomRequest_old(self, delta):
        units = delta / 120  # 调整单位转换，通常一个滚轮步长是120
        scale = 10
        increment = scale * units

        # Step 1: 获取当前视图中心的图像坐标
        currentCenterX = self.scrollBars[Qt.Horizontal].value() + self.scroll.viewport().width() / 2
        currentCenterY = self.scrollBars[Qt.Vertical].value() + self.scroll.viewport().height() / 2

        # Step 2: 更新缩放级别
        oldZoomValue = self.zoomWidget.value()
        newZoomValue = oldZoomValue + increment
        self.setZoom(newZoomValue)

        # Step 3: 计算新的缩放比例
        oldScale = oldZoomValue / 100.0
        newScale = newZoomValue / 100.0

        # Step 4: 根据新的缩放级别重新计算并设置滚动条位置
        newCenterX = (currentCenterX * newScale / oldScale) - self.scroll.viewport().width() / 2
        newCenterY = (currentCenterY * newScale / oldScale) - self.scroll.viewport().height() / 2
        self.scrollBars[Qt.Horizontal].setValue(int(newCenterX))
        self.scrollBars[Qt.Vertical].setValue(int(newCenterY))


    def zoomRequest(self, delta):
        """
        处理缩放请求。
        :param delta: 缩放的增量，通常来自鼠标滚轮事件。
        """
        units = delta / 120  # 通常一个滚轮步长是120
        scale_increment = 10
        increment = scale_increment * units

        # Step 1: 更新缩放级别
        oldZoomValue = self.zoomWidget.value()
        newZoomValue = oldZoomValue + increment

        # Clamp newZoomValue to 10 - 500
        newZoomValue = max(10, min(newZoomValue, 500))
        # print(f"Old Zoom Value: {oldZoomValue}, New Zoom Value: {newZoomValue}")

        # Step 2: 计算新的缩放比例
        oldScale = oldZoomValue / 100.0
        newScale = newZoomValue / 100.0
        # print(f"Old Scale: {oldScale}, New Scale: {newScale}")

        # Step 3: 确定缩放中心点
        if self.current_center is not None:
            # 使用当前的中心点（图像坐标）
            center_x, center_y = self.current_center.x(), self.current_center.y()
            # print(f"Using current center point: ({center_x}, {center_y})")
        else:
            # 使用视口中心点转换为图像坐标
            h_scroll = self.scroll.horizontalScrollBar().value()
            v_scroll = self.scroll.verticalScrollBar().value()
            viewport_width = self.scroll.viewport().width()
            viewport_height = self.scroll.viewport().height()
            center_x = (h_scroll + viewport_width / 2) / oldScale
            center_y = (v_scroll + viewport_height / 2) / oldScale
            # print(f"Using viewport center point: ({center_x}, {center_y})")

        # Step 4: 更新缩放级别
        self.setZoom(newZoomValue)

        # Step 5: 计算新的滚动条位置以保持缩放中心不变
        # 新的缩放比例下，计算中心点的缩放后坐标
        scaled_x = center_x * newScale
        scaled_y = center_y * newScale

        # 获取视口大小
        viewport_width = self.scroll.viewport().width()
        viewport_height = self.scroll.viewport().height()

        # 计算新的滚动条位置，使中心点位于视图中央
        newScrollX = scaled_x - viewport_width / 2
        newScrollY = scaled_y - viewport_height / 2

        # print(f"New Scroll X before clamping: {newScrollX}, New Scroll Y before clamping: {newScrollY}")

        # 限制滚动条的位置在有效范围内
        newScrollX = max(0, min(int(round(newScrollX)), self.scroll.horizontalScrollBar().maximum()))
        newScrollY = max(0, min(int(round(newScrollY)), self.scroll.verticalScrollBar().maximum()))
        # print(f"New Scroll X after clamping: {newScrollX}, New Scroll Y after clamping: {newScrollY}")

        # 设置滚动条的位置
        self.scroll.horizontalScrollBar().setValue(newScrollX)
        self.scroll.verticalScrollBar().setValue(newScrollY)


        # print(f"Scroll bar positions set to X: {newScrollX}, Y: {newScrollY}")


    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile_ori(self, filePath = None) :
        """Load the specified file, or the last opened file if None."""
        currIndex = 1
        filename_1 = None
        filename_2 = None
        if filePath:
            currIndex = self.mImgList.index(filePath)
        if currIndex + 1 < len(self.mImgList):
            filename_1 = self.mImgList[currIndex + 1]
            filename_2 = self.mImgList[currIndex + 2]

        self.resetState()
        self.canvas.setEnabled(False)
        self.canvas_1.setEnabled(False)
        self.canvas_2.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get('filename')

        print(filePath)
        unicodeFilePath = ustr(filePath)
        unicodeFilePath_1 = ustr(filename_1)
        unicodeFilePath_2 = ustr(filename_2)

        print(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            index = self.mImgList.index(unicodeFilePath)
            fileWidgetItem = self.fileListWidget.item(index)
            fileWidgetItem.setSelected(True)

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)

                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.imageData_1 = read(unicodeFilePath_1, None)
                self.imageData_2 = read(unicodeFilePath_2, None)
                self.labelFile = None

            image = QImage.fromData(self.imageData)
            image_1 = QImage.fromData(self.imageData_1)
            image_2 = QImage.fromData(self.imageData_2)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
        
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            self.image = image
            self.filePath = unicodeFilePath
            self.filePath_1 = unicodeFilePath_1
            self.filePath_2 = unicodeFilePath_2
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            self.canvas_1.loadPixmap(QPixmap.fromImage(image_1))
            self.canvas_2.loadPixmap(QPixmap.fromImage(image_2))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes,self.labelFile.shapes_1,self.labelFile.shapes_2)
            self.setClean()
            self.canvas.setEnabled(True)
            self.canvas_1.setEnabled(True)
            self.canvas_2.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            # Label xml file and show bound box according to its filename
            if self.usingPascalVocFormat is True:
                if self.defaultSaveDir is not None:
                    basename = os.path.basename(
                        os.path.splitext(self.filePath)[0]) + XML_EXT
                    xmlPath = os.path.join(self.defaultSaveDir, basename)
                    basename_1 = os.path.basename(
                        os.path.splitext(self.filePath_1)[0]) + XML_EXT
                    xmlPath_1 = os.path.join(self.defaultSaveDir, basename_1)
                    basename_2 = os.path.basename(
                        os.path.splitext(self.filePath_2)[0]) + XML_EXT
                    xmlPath_2 = os.path.join(self.defaultSaveDir, basename_2)
                    self.loadPascalXMLByFilename(xmlPath, xmlPath_1, xmlPath_2)
                else:
                    xmlPath = filePath.split(".")[0] + XML_EXT
                    xmlPath_1 = filename_1.split(".")[0] + XML_EXT
                    xmlPath_2 = filename_2.split(".")[0] + XML_EXT
                    if os.path.isfile(xmlPath):
                        self.loadPascalXMLByFilename(xmlPath, xmlPath_1, xmlPath_2)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
                self.labelList.setCurrentItem(self.labelList.item(0))
                # self.labelList.setItemSelected(self.labelList.item(self.labelList.count()-1), True)

            self.canvas.setFocus(True)
            self.canvas_1.setFocus(True)
            self.canvas_2.setFocus(True)
            return True
        return False




    def loadFile(self, groupIndex=None, filename=None) :
        """Load a group of three images based on group index or a specific filename."""
        if filename is not None:
            # 根据 filename 找到对应的 groupIndex
            groupIndex = None
            for index, (prefix, image_paths) in enumerate(self.mImgList):
                if filename in image_paths:
                    groupIndex = index
                    break
            if groupIndex is None:
                print("文件未找到在任何组中。")
                return False
        
        if groupIndex is None:
            groupIndex = 0  # 默认加载第一个组

        if groupIndex is not None and filename is not None:
            self.updateProgressDisplay()

        if not isinstance(groupIndex, int):
            print("groupIndex 必须是整数。")
            return False

        if groupIndex < 0 or groupIndex >= len(self.mImgList):
            print("组索引超出范围。")
            return False
        
        # 选择列表中的当前项
        current_item = self.fileListWidget.item(groupIndex)
        if current_item:
            self.fileListWidget.setCurrentItem(current_item)  # 设置选中
            self.fileListWidget.scrollToItem(current_item, QAbstractItemView.PositionAtCenter)  # 确保可见

        prefix, image_paths = self.mImgList[groupIndex]
        if len(image_paths) != 3:
            print(f"组 {prefix} 中的文件数量不为3，已跳过。")
            return False

        # 重置状态
        self.resetState()
        self.canvas.setEnabled(False)
        self.canvas_1.setEnabled(False)
        self.canvas_2.setEnabled(False)

        # 加载所有三张图像
        images = []
        for img_path in image_paths:
            image_data = read(img_path, None)
            if image_data is None:
                self.errorMessage(u'错误', f"无法读取图像文件: {img_path}")
                return False
            image = QImage.fromData(image_data)
            if image.isNull():
                self.errorMessage(u'错误', f"无效的图像文件: {img_path}")
                return False
            images.append(image)

        # 设置图像和路径
        self.image = images[0]
        self.image_1 = images[1]
        self.image_2 = images[2]
        filePath = image_paths[0]
        filePath_1 = image_paths[1]
        filePath_2 = image_paths[2]
        self.filePath = filePath
        self.filePath_1 = filePath_1
        self.filePath_2 = filePath_2
        # 加载图像到画布
        self.canvas.loadPixmap(QPixmap.fromImage(self.image))
        self.canvas_1.loadPixmap(QPixmap.fromImage(self.image_1))
        self.canvas_2.loadPixmap(QPixmap.fromImage(self.image_2))


        if self.labelFile:
            self.loadLabels(self.labelFile.shapes,self.labelFile.shapes_1,self.labelFile.shapes_2)
        self.setClean()
        self.canvas.setEnabled(True)
        self.canvas_1.setEnabled(True)
        self.canvas_2.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.addRecentFile(self.filePath)
        self.toggleActions(True)

        # Label xml file and show bound box according to its filename
        if self.usingPascalVocFormat is True:
            if self.defaultSaveDir is not None:
                basename = os.path.basename(
                    os.path.splitext(filePath)[0]) + XML_EXT
                xmlPath = os.path.join(self.defaultSaveDir, basename)
                basename_1 = os.path.basename(
                    os.path.splitext(filePath_1)[0]) + XML_EXT
                xmlPath_1 = os.path.join(self.defaultSaveDir, basename_1)
                basename_2 = os.path.basename(
                    os.path.splitext(filePath_2)[0]) + XML_EXT
                xmlPath_2 = os.path.join(self.defaultSaveDir, basename_2)
                self.loadPascalXMLByFilename(xmlPath, xmlPath_1, xmlPath_2)
            else:
                xmlPath = filePath.split(".")[0] + XML_EXT
                xmlPath_1 = filePath_1.split(".")[0] + XML_EXT
                xmlPath_2 = filePath_2.split(".")[0] + XML_EXT
                if os.path.isfile(xmlPath):
                    self.loadPascalXMLByFilename(xmlPath, xmlPath_1, xmlPath_2)

        self.setWindowTitle(__appname__ + ' ' + filePath)

        # Default : select last item if there is at least one item
        if self.labelList.count():
            self.labelList.setCurrentItem(self.labelList.item(0))
            # self.labelList.setItemSelected(self.labelList.item(self.labelList.count()-1), True)

        self.canvas.setFocus(True)
        self.canvas_1.setFocus(True)
        self.canvas_2.setFocus(True)

        # At the end of the loadFile method, after successfully loading the image, add:
        self.updateCoordinatesDisplay()
        self.updateAnnotationCount()

        return True

    


    def resizeEvent(self, event):
        # print("resizeEvent")
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)


    def paintCanvas(self):
        # print("paintCanvas")
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoom_level  # 使用 self.zoom_level 而不是 zoomWidget.value()
        self.canvas.adjustSize()
        self.canvas.update()
        self.canvas_1.scale = 0.01 * self.zoom_level  # 使用 self.zoom_level 而不是 zoomWidget.value()
        self.canvas_1.adjustSize()
        self.canvas_1.update()
        self.canvas_2.scale = 0.01 * self.zoom_level  # 使用 self.zoom_level 而不是 zoomWidget.value()
        self.canvas_2.adjustSize()
        self.canvas_2.update()


    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        print("scaleFitWindow")
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        print("scaleFitWidth")
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        s = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            s['filename'] = self.filePath if self.filePath else ''
        else:
            s['filename'] = ''

        s['window/size'] = self.size()
        s['window/position'] = self.pos()
        s['window/state'] = self.saveState()
        s['line/color'] = self.lineColor
        s['fill/color'] = self.fillColor
        s['recentFiles'] = self.recentFiles
        s['advanced'] = not self._beginner
        if self.defaultSaveDir is not None and len(self.defaultSaveDir) > 1:
            s['savedir'] = ustr(self.defaultSaveDir)
        else:
            s['savedir'] = ""

        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            s['lastOpenDir'] = self.lastOpenDir
        else:
            s['lastOpenDir'] = ""

    ## User Dialogs ##

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages_ori(self, folderPath):
        extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relatviePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relatviePath))
                    images.append(path)
        images.sort(key=lambda x: x.lower())
        return images

    def scanAllImages(self, folderPath):
        extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        image_files = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    filepath = os.path.join(root, file)
                    image_files.append(os.path.abspath(filepath))
        image_files.sort(key=lambda x: x.lower())

        # 使用正则表达式提取前缀和年份
        pattern = re.compile(r'(.+?)_(\d{4})\.(jpg|jpeg|png|bmp)$', re.IGNORECASE)
        groups = defaultdict(list)

        for filepath in image_files:
            filename = os.path.basename(filepath)
            match = pattern.match(filename)
            if match:
                prefix = match.group(1)
                year = int(match.group(2))
                groups[prefix].append( (year, filepath) )
            else:
                # 如果文件不符合模式，可以选择忽略或单独处理
                print(f"忽略不符合模式的文件: {filename}")

        # 按年份排序每个组
        grouped_images = []
        for prefix, files in groups.items():
            sorted_files = sorted(files, key=lambda x: x[0])  # 按年份升序
            sorted_filepaths = [f[1] for f in sorted_files]
            if len(sorted_filepaths) == 3:
                grouped_images.append( (prefix, sorted_filepaths) )
            else:
                print(f"组 {prefix} 中的文件数量不为3，已跳过。")

        # 按前缀排序组
        grouped_images.sort(key=lambda x: x[0].lower())

        return grouped_images



    def changeSavedir(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotation(self, _value=False):
        if self.filePath is None:
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.usingPascalVocFormat:
            filters = "Open Annotation XML file (%s)" % \
                      ' '.join(['*.xml'])
            filename = QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters)
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDi_ori(self, _value=False):
        if not self.mayContinue():
            return

        path = os.path.dirname(self.filePath)\
            if self.filePath else '.'

        if self.lastOpenDir is not None and len(self.lastOpenDir) > 1:
            path = self.lastOpenDir

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                     '%s - Open Directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                     | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.lastOpenDir = dirpath

        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()
        self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)


    def openDir(self, _value=False):
        if not self.mayContinue():
            return

        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'

        if self.lastOpenDir and len(self.lastOpenDir) > 1:
            path = self.lastOpenDir

        dirpath = ustr(QFileDialog.getExistingDirectory(
            self,
            '%s - Open Directory' % __appname__,
            path,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        ))

        if dirpath and len(dirpath) > 1:
            self.lastOpenDir = dirpath

        self.dirname = dirpath
        self.filePath = None
        self.fileListWidget.clear()

        # 扫描并分组图像
        self.mImgList = self.scanAllImages(dirpath)

        # 如果没有有效的组，提示用户
        if not self.mImgList:
            QMessageBox.information(self, "信息", "未找到符合条件的图像组。")
            return

        # 将前缀添加到文件列表中
        for group in self.mImgList:
            prefix = group[0]
            display_name = prefix.title()  # 根据需要格式化
            item = QListWidgetItem(display_name)
            self.fileListWidget.addItem(item)


        # 加载第一个组
        self.loadFile(0)

        # # 将所有图像添加到文件列表中
        # for imgPath in self.mImgList:
        #     pass  # 由于只显示前缀，实际图像加载在 loadFile 中处理

        return

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
         if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile is not None:
                    self.labelFile.toggleVerify()
            if self.labelFile is not None:
                self.canvas.verified = True
            self.paintCanvas()
            self.saveFile()
            self.Next()


    def openPrevImg_ori(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg_ori(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.autoSaving is True and self.defaultSaveDir is not None:
            if self.dirty is True: 
                self.dirty = False
                self.canvas.verified = True               
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]

        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)


    def openPrevImg(self, _value=False):
        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        if self.filePath is None:
            return

        # 查找当前文件属于哪个组
        currIndex = None
        for i, (prefix, image_paths) in enumerate(self.mImgList):
            if self.filePath in image_paths:
                currIndex = i
                break

        if currIndex is None:
            print("错误：未能在mImgList中找到当前文件")
            return

        # 计算前一个索引
        prevIndex = currIndex - 1
        if prevIndex < 0:  # 允许循环到最后一个
            prevIndex = len(self.mImgList) - 1

        # 加载上一组图像
        self.loadFile(groupIndex=prevIndex)


    def openNextImg(self, _value=False):
        # 自动保存当前标注
        if self.autoSaving and self.defaultSaveDir:
            if self.dirty:
                self.dirty = False
                self.canvas.verified = True
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 查找当前文件属于哪个组
        currIndex = None
        for i, (prefix, image_paths) in enumerate(self.mImgList):
            if self.filePath in image_paths:
                currIndex = i
                break

        if currIndex is None:
            print("错误：未能在mImgList中找到当前文件")
            return

        # 计算下一个索引
        nextIndex = currIndex + 1
        if nextIndex >= len(self.mImgList):  # 允许循环回到第一个
            nextIndex = 0

        # 加载下一组图像
        self.loadFile(groupIndex=nextIndex)


    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def saveFile(self, _value=False):
        # 使用默认保存路径
        file_paths_and_canvases = [
            (self.filePath, self.canvas),
            (self.filePath_1, self.canvas_1),
            (self.filePath_2, self.canvas_2)
        ]
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            for file_path, canvas in file_paths_and_canvases:
                if file_path:
                    imgFileName = os.path.basename(file_path)
                    savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
                    savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                    # 调用 _saveFile 并传递 canvas 参数
                    self._saveFile(savedPath, canvas)
        else:
            # 使用自定义路径
            for file_path, canvas in file_paths_and_canvases:
                if file_path:
                    imgFileDir = os.path.dirname(file_path)
                    imgFileName = os.path.basename(file_path)
                    savedFileName = os.path.splitext(imgFileName)[0] + XML_EXT
                    savedPath = os.path.join(imgFileDir, savedFileName)
                    # 选择是否保存到默认路径或自定义路径
                    self._saveFile(savedPath if self.labelFile else self.saveFileDialog(), canvas)
        self.updateProgressDisplay()

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            return dlg.selectedFiles()[0]
        return ''

    def _saveFile(self, annotationFilePath, canvas):
        if annotationFilePath and self.saveLabels(annotationFilePath, canvas):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = u'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, u'Attention', msg, yes | no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        print("color1")
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
        print("color2")
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape_old(self):
        # 检查每个canvas的选中shape并删除
        print("delete")
        for canvas in [self.canvas, self.canvas_1, self.canvas_2]:
            try:
            # 尝试删除选中的形状
                selected_shape = canvas.deleteSelected()
                if selected_shape:  # 如果删除成功且有选中的形状
                    self.remLabel(selected_shape)

            except Exception as e:
                # 捕获异常并跳过该canvas的删除操作
                print(f"Error deleting from {canvas}: {e}")
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def deleteSelectedShape(self):
        # 检查每个canvas的选中shape并删除
        print("delete")
        
        # 确保只从当前选中的画布删除
        selected_shapes = []
        for canvas in [self.canvas, self.canvas_1, self.canvas_2]:
            selected_shape = canvas.selectedShape
            if selected_shape:
                selected_shapes.append((canvas, selected_shape))

        # 执行删除
        for canvas, shape in selected_shapes:
            try:
                # 尝试删除选中的形状
                canvas.deleteSelected()  # 删除当前画布的选中形状
                self.remLabel(shape)  # 删除对应的标签
            except Exception as e:
                # 捕获异常并跳过该canvas的删除操作
                print(f"Error deleting from {canvas}: {e}")

        self.setDirty()
        self.updateAnnotationCount()

        # 更新动作的可用状态
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)


    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.lablHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath, xmlPath_1, xmlPath_2):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        if not os.path.exists(xmlPath):
            if os.path.exists(xmlPath_1):
                shutil.copy(xmlPath_1, xmlPath)
                print("Copying from 1 to 0")
            elif os.path.exists(xmlPath_2):
                shutil.copy(xmlPath_2, xmlPath)
                print("Copying from 2 to 0")

        if not os.path.exists(xmlPath_1):
            if os.path.exists(xmlPath):
                shutil.copy(xmlPath, xmlPath_1)
                print("Copying from 0 to 1")
            elif os.path.exists(xmlPath_2):
                shutil.copy(xmlPath_2, xmlPath_1)
                print("Copying from 2 to 1")
        
        if not os.path.exists(xmlPath_2):
            if os.path.exists(xmlPath):
                shutil.copy(xmlPath, xmlPath_2)
                print("Copying from 0 to 2")
            elif os.path.exists(xmlPath_1):
                shutil.copy(xmlPath_1, xmlPath_2)
                print("Copying from 1 to 2")

        tVocParseReader = DotaReader(xmlPath)
        tVocParseReader_1 = DotaReader(xmlPath_1)
        tVocParseReader_2 = DotaReader(xmlPath_2)
        shapes = tVocParseReader.getShapes()
        shapes_1 = tVocParseReader_1.getShapes()
        shapes_2 = tVocParseReader_2.getShapes()
        self.loadLabels(shapes, shapes_1, shapes_2)
        self.canvas.verified = tVocParseReader.verified
        self.canvas_1.verified = tVocParseReader_1.verified
        self.canvas_2.verified = tVocParseReader_2.verified

    def Next_ori(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.autoSaving is True and self.defaultSaveDir is not None:
            if self.dirty is True:
                self.dirty = False
                self.canvas.verified = True
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        #filename_1 = None
        if self.filePath is None:
            filename = self.mImgList[0]
            #filename_1 = self.mImgList[1]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 3 < len(self.mImgList):
                filename = self.mImgList[currIndex + 3]
                #filename_1 = self.mImgList[currIndex + 2]

        if filename:
            self.loadFile(filename)

            #self.loadFile(filename_1)

    def Next(self, _value=False):
        # 自动保存当前标注
        if self.autoSaving and self.defaultSaveDir:
            if self.dirty:
                self.dirty = False
                self.canvas.verified = True
                self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 查找当前文件属于哪个组
        currIndex = None
        for i, (prefix, image_paths) in enumerate(self.mImgList):
            if self.filePath in image_paths:
                currIndex = i
                break

        if currIndex is None:
            print("错误：未能在mImgList中找到当前文件")
            return

        # 计算下一个索引
        nextIndex = currIndex + 1
        if nextIndex >= len(self.mImgList):  # 超过范围则循环回到第一个
            nextIndex = 0

        # 加载下一组图像
        self.loadFile(groupIndex=nextIndex)



    def Back(self):
        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 3 >= 0:
            filename = self.mImgList[currIndex - 3]
            if filename:
                self.loadFile(filename)

    def add_2010(self):
        # 获取当前选中的标注
        selected_shape = self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected in the first image.")
            return

        # 克隆标注对象
        cloned_shape = selected_shape.copy()

        # 将克隆的标注添加到第一个 Canvas 中
        self.canvas.addShape(cloned_shape)

        # 同步更新标签列表
        item = HashableQListWidgetItem(cloned_shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = cloned_shape
        self.shapesToItems[cloned_shape] = item
        self.labelList.addItem(item)

        # 更新画布显示
        self.canvas.update()
        self.actions.save.setEnabled(True)
        print(f"Copied selected shape '{selected_shape.label}' to the first image and added label to the list.")
        self.updateAnnotationCount()

    def add_2015(self):
        # 获取当前选中的标注
        selected_shape = self.canvas.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected in the first image.")
            return

        # 克隆标注对象
        cloned_shape = selected_shape.copy()

        # 同步更新标签列表
        item = HashableQListWidgetItem(cloned_shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = cloned_shape
        self.shapesToItems[cloned_shape] = item
        self.labelList.addItem(item)

        # 将克隆的标注添加到第一个Canvas中
        self.canvas_1.addShape(cloned_shape)

        # 更新画布显示
        self.canvas_1.update()
        self.actions.save.setEnabled(True)
        print(f"Copied selected shape '{selected_shape.label}' to both the second and third images.")
        self.updateAnnotationCount()

    def add_2020(self):
        # 获取当前选中的标注
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape
        if selected_shape is None:
            print("No shape selected in the first image.")
            return

        # 克隆标注对象
        cloned_shape = selected_shape.copy()

        # 同步更新标签列表
        item = HashableQListWidgetItem(cloned_shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = cloned_shape
        self.shapesToItems[cloned_shape] = item
        self.labelList.addItem(item)

        # 将克隆的标注添加到第一个Canvas中
        self.canvas_2.addShape(cloned_shape)

        # 更新画布显示
        self.canvas_2.update()
        self.actions.save.setEnabled(True)
        print(f"Copied selected shape '{selected_shape.label}' to both the second and third images.")
        self.updateAnnotationCount()

    def add_in_mid(self):
        # 获取水平和垂直滚动条的对象
        horizontal_scrollbar = self.scroll.horizontalScrollBar()  # 水平滚动条self.scroll.horizontalScrollBar()
        vertical_scrollbar = self.scroll.verticalScrollBar()  # 垂直滚动条

        # 获取滚动条的实际长度（可视区域的大小）
        horizontal_length = horizontal_scrollbar.pageStep()  # 水平滚动条的长度
        vertical_length = vertical_scrollbar.pageStep()  # 垂直滚动条的长度

        # 计算滚动条的可视区域中心位置
        center_x = horizontal_length / 2  # 水平滚动条中心
        center_y = vertical_length / 2  # 垂直滚动条中心

        print(f"Horizontal center: {center_x}, Vertical center: {center_y}")

        # 创建一个固定大小的标签（Shape）
        label_size = 50  # 假定标签的固定大小为 50x50 像素

        # 创建一个Shape对象，并初始化标签和其它参数
        label = Shape(label="embankment_dam", difficult=False)

        # 设置标签的中心位置
        label.center = QPointF(center_x, center_y)

        # 假设创建一个矩形形状，我们需要向 shape 对象添加 4 个点来表示该矩形
        # 这里将矩形的宽度和高度设置为 label_size
        label.addPoint(QPointF(center_x - label_size / 2, center_y - label_size / 2))  # 左上角
        label.addPoint(QPointF(center_x + label_size / 2, center_y - label_size / 2))  # 右上角
        label.addPoint(QPointF(center_x + label_size / 2, center_y + label_size / 2))  # 右下角
        label.addPoint(QPointF(center_x - label_size / 2, center_y + label_size / 2))  # 左下角

        # 将标签添加到画布
        self.canvas.addShape(label)

        # 同步更新标签列表
        item = HashableQListWidgetItem(label.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.itemsToShapes[item] = label
        self.shapesToItems[label] = item
        self.labelList.addItem(item)

        # 更新画布显示
        self.canvas.update()
        self.actions.save.setEnabled(True)
        print(f"Added centered label '{label.label}' at the center of the scrollable area.")

    def unify_all_label(self):
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected in the first image.")
            return

        selected_label = selected_shape.label
        selected_rect = selected_shape.boundingRect()

        # 定义一个函数来处理单个 Canvas 上的标签更新
        def update_labels_in_canvas(target_canvas):
            for shape in target_canvas.shapes:
                target_rect = shape.boundingRect()
                # 检查重叠：这里使用 `intersects` 判断边界矩形是否重叠
                if selected_rect.intersects(target_rect):
                    print("overlap exist")
                    print(shape.label)
                    shape.label = selected_label  # 统一名称
                    print(shape.label)
                    # 同步更新标签列表中的对应项
                    item = self.shapesToItems.get(shape)
                    if item:
                        item.setText(selected_label)
                        print(f"Updated labelList item to '{selected_label}'")

        # 处理目标 canvas
        update_labels_in_canvas(self.canvas)
        update_labels_in_canvas(self.canvas_1)
        update_labels_in_canvas(self.canvas_2)

        # 更新画布显示
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()
        self.actions.save.setEnabled(True)
        print(f"Updated overlapping labels in target canvases to '{selected_label}'")


    def delete_and_next(self, _value=False):
        """
        删除当前选中的标签及重叠标签，保存并跳转到下一组图像。
        """
        # 步骤 1: 删除选中的标签及其重叠标签
        self.delete_all_label()

        # 步骤 2: 保存当前图像的标注
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # 如果标注文件尚未创建，则创建并保存
                self.saveFile()
                if self.labelFile is not None:
                    self.labelFile.toggleVerify()
            if self.labelFile is not None:
                self.canvas.verified = True
            self.paintCanvas()
            self.saveFile()

        # 步骤 3: 跳转到下一组图像
        self.Next()


    def delete_all_label(self):
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for deletion.")
            return

        selected_label = selected_shape.label
        selected_rect = selected_shape.boundingRect()


        # 定义一个函数来删除 canvas 上的重叠标签
        def delete_labels_in_canvas(target_canvas):
            shapes_to_delete = []  # 存储需要删除的形状
            for shape in target_canvas.shapes:
                target_rect = shape.boundingRect()
                # 检查重叠，若重叠则将 shape 加入删除列表
                if selected_rect.intersects(target_rect) and shape.label == selected_label:
                    shapes_to_delete.append(shape)

            # 从 canvas 和 labelList 中删除重叠标签
            for shape in shapes_to_delete:
                target_canvas.shapes.remove(shape)
                item = self.shapesToItems.pop(shape, None)  # 删除 shapesToItems 映射
                if item:
                    self.labelList.takeItem(self.labelList.row(item))  # 从标签列表中删除
                    print(f"Deleted labelList item '{selected_label}'")

        # 删除当前选中的标签以及在其他 canvas 中重叠的标签
        delete_labels_in_canvas(self.canvas)
        delete_labels_in_canvas(self.canvas_1)
        delete_labels_in_canvas(self.canvas_2)

        # 更新画布显示
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()
        self.actions.save.setEnabled(True)
        print(f"Deleted selected label '{selected_label}' and overlapping labels in all canvases.")

    def remove(self):
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for deletion.")
            return

        selected_label = selected_shape.label
        selected_rect = selected_shape.boundingRect()

        # 定义一个函数来删除 canvas 上的重叠标签
        def hide_labels_in_canvas(target_canvas):
            shapes_to_delete = []  # 存储需要删除的形状
            for shape in target_canvas.shapes:
                target_rect = shape.boundingRect()
                # 检查重叠，若重叠则将 shape 加入删除列表
                if selected_rect.intersects(target_rect) and shape.label == selected_label:
                    shapes_to_delete.append(shape)

            # 从 canvas 和 labelList 中删除重叠标签
            for shape in shapes_to_delete:
                item = self.shapesToItems.get(shape, None)  # 删除 shapesToItems 映射
                item.setHidden(True)
                #shape.setHidden(True)
                if item:
                    self.labelList.takeItem(self.labelList.row(item))  # 从标签列表中删除
                    print(f"Deleted labelList item '{selected_label}'")

        # 删除当前选中的标签以及在其他 canvas 中重叠的标签
        hide_labels_in_canvas(self.canvas)
        hide_labels_in_canvas(self.canvas_1)
        hide_labels_in_canvas(self.canvas_2)

        # 更新画布显示
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()
        self.actions.save.setEnabled(True)

    def select_all(self):
        # 获取当前选中的标注框
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected.")
            return

        # 获取选中框的位置范围
        selected_rect = selected_shape.boundingRect()

        # 定义一个函数来选择指定 canvas 上的重叠标注框
        def select_overlapping_shapes_in_canvas(target_canvas):
            for shape in target_canvas.shapes:
                target_rect = shape.boundingRect()
                # 检查是否与已选标注框重叠
                if selected_rect.intersects(target_rect):
                    # 标记为选中
                    shape.selected = True
                    item = self.shapesToItems.get(shape)
                    if item:
                        item.setSelected(True)  # 在标签列表中同步选择

        # 在所有 canvas 中查找和选中重叠的标注框
        select_overlapping_shapes_in_canvas(self.canvas)
        select_overlapping_shapes_in_canvas(self.canvas_1)
        select_overlapping_shapes_in_canvas(self.canvas_2)

        # 更新画布显示，以显示选中效果
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()
        self.actions.save.setEnabled(True)
        print("Selected all overlapping shapes in all canvases.")

    def delete_2010(self):
        # 获取当前选中框
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected.")
            return

        # 获取选中框的范围
        selected_rect = selected_shape.boundingRect()

        # 在 canvas 中查找并删除重叠框
        shapes_to_delete = []
        for shape in self.canvas.shapes:
            target_rect = shape.boundingRect()
            # 检查重叠：如果有重叠，添加到删除列表
            if selected_rect.intersects(target_rect):
                shapes_to_delete.append(shape)

        # 删除找到的重叠框
        for shape in shapes_to_delete:
            self.canvas.shapes.remove(shape)
            # 同步移除 shapesToItems 中对应的项
            item = self.shapesToItems.pop(shape, None)
            if item:
                self.labelList.takeItem(self.labelList.row(item))

        # 更新画布显示
        self.canvas.update()
        self.actions.save.setEnabled(True)
        print("Deleted overlapping shapes in canvas based on selection in canvas_1 or canvas_2.")
        self.updateAnnotationCount()

    def delete_2015(self):
        # 获取当前选中框
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected.")
            return

        # 获取选中框的范围
        selected_rect = selected_shape.boundingRect()

        # 在 canvas 中查找并删除重叠框
        shapes_to_delete = []
        for shape in self.canvas_1.shapes:
            target_rect = shape.boundingRect()
            # 检查重叠：如果有重叠，添加到删除列表
            if selected_rect.intersects(target_rect):
                shapes_to_delete.append(shape)

        # 删除找到的重叠框
        for shape in shapes_to_delete:
            self.canvas_1.shapes.remove(shape)
            # 同步移除 shapesToItems 中对应的项
            item = self.shapesToItems.pop(shape, None)
            if item:
                self.labelList.takeItem(self.labelList.row(item))

        # 更新画布显示
        self.canvas_1.update()
        self.actions.save.setEnabled(True)
        print("Deleted overlapping shapes in canvas based on selection in canvas_1 or canvas_2.")
        self.updateAnnotationCount()

    def delete_2020(self):
        # 获取当前选中框
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected.")
            return

        # 获取选中框的范围
        selected_rect = selected_shape.boundingRect()

        # 在 canvas 中查找并删除重叠框
        shapes_to_delete = []
        for shape in self.canvas_2.shapes:
            target_rect = shape.boundingRect()
            # 检查重叠：如果有重叠，添加到删除列表
            if selected_rect.intersects(target_rect):
                shapes_to_delete.append(shape)

        # 删除找到的重叠框
        for shape in shapes_to_delete:
            self.canvas_2.shapes.remove(shape)
            # 同步移除 shapesToItems 中对应的项
            item = self.shapesToItems.pop(shape, None)
            if item:
                self.labelList.takeItem(self.labelList.row(item))

        # 更新画布显示
        self.canvas_2.update()
        self.actions.save.setEnabled(True)
        print("Deleted overlapping shapes in canvas based on selection in canvas_1 or canvas_2.")
        self.updateAnnotationCount()

    def modify_labels_in_canvas(self, selected_shape, new_label):
        """修改所有画布中与选定标签重叠的标签名称"""
        selected_rect = selected_shape.boundingRect()

        # 遍历每个画布进行修改
        def modify_in_target_canvas(target_canvas):
            for shape in target_canvas.shapes:
                target_rect = shape.boundingRect()
                if selected_rect.intersects(target_rect) and shape.label == selected_shape.label:
                    shape.label = new_label  # 修改标签名称
                    # 更新标签列表
                    item = self.shapesToItems.get(shape)
                    if item:
                        item.setText(new_label)
                    print(f"Modified label '{selected_shape.label}' to '{new_label}' in canvas.")

        # 不依赖于选中的画布，遍历所有三个画布进行修改
        modify_in_target_canvas(self.canvas)
        modify_in_target_canvas(self.canvas_1)
        modify_in_target_canvas(self.canvas_2)

        # 更新画布显示
        self.canvas.update()
        self.canvas_1.update()
        self.canvas_2.update()



    def embankment(self):
        """修改选定标签及其在三个画布中重叠标签的名称为指定的标签名称"""
        print("edit selected labels")
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for modification.")
            return

        new_label = "embankment_dam"
        
        # 使用通用函数修改重叠标签
        self.modify_labels_in_canvas(selected_shape, new_label)

    def Barrage(self):
        """修改选定标签及其在三个画布中重叠标签的名称为指定的标签名称"""
        print("edit selected labels")
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for modification.")
            return

        new_label = "Barrage_dam"

        # 使用通用函数修改重叠标签
        self.modify_labels_in_canvas(selected_shape, new_label)

    def gravity(self):
        """修改选定标签及其在三个画布中重叠标签的名称为指定的标签名称"""
        print("edit selected labels")
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for modification.")
            return

        new_label = "gravity_dam"

        # 使用通用函数修改重叠标签
        self.modify_labels_in_canvas(selected_shape, new_label)

    def arch(self):
        """修改选定标签及其在三个画布中重叠标签的名称为指定的标签名称"""
        print("edit selected labels")
        selected_shape = self.canvas.selectedShape or self.canvas_1.selectedShape or self.canvas_2.selectedShape
        if selected_shape is None:
            print("No shape selected for modification.")
            return

        new_label = "arch_dam"

        # 使用通用函数修改重叠标签
        self.modify_labels_in_canvas(selected_shape, new_label)


    # 显示不同年份的标签
    def display_year_labels(self):
        # 清空显示区域
        self.labelyears.clear()

        # 目标 canvas 列表，按固定顺序排列
        canvases = [
            (self.canvas, "2010_"),
            (self.canvas_1, "2015_"),
            (self.canvas_2, "2020_")
        ]

        # 查找当前选中的 shape（可以在任意 canvas 中）
        selected_shape = (
                self.canvas.selectedShape or
                self.canvas_1.selectedShape or
                self.canvas_2.selectedShape
        )

        # 如果没有选中标签，直接返回
        if selected_shape is None:
            return

        # 获取选中标签的边界矩形
        selected_rect = selected_shape.boundingRect()

        # 遍历各 canvas，按固定顺序查找重叠标签
        for canvas, prefix in canvases:
            found = False
            # 查找与选中标签重叠的标签
            for shape in canvas.shapes:
                if selected_rect.intersects(shape.boundingRect()):
                    self.labelyears.addItem(f"{prefix}{shape.label}")
                    found = True
                    break

            # 如果当前 canvas 中没有找到重叠标签，则显示空标签项
            if not found:
                self.labelyears.addItem(f"{prefix}(无标签)")

    def openNextChangedLabeledImg(self, _value=False):
        """
        跳转到下一组有标签但是标签有变化的图像（考虑所有三个视图）。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 如果没有已打开的文件，尝试找到第一张有标签的文件
        if self.filePath is None:
            for i in range(len(self.mImgList)):
                if self.has_labels(i) and self.has_label_changes(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到标签变化的图像组。')
            return

        # 查找当前组索引
        current_group_index = next(
            (i for i, group in enumerate(self.mImgList) if self.filePath in group[1]),
            -1
        )

        # 从当前组索引的下一个开始查找
        for i in range(current_group_index + 1, len(self.mImgList)):
            if self.has_labels(i) and self.has_label_changes(i):
                self.loadFile(i)
                return

        # 没有找到，提示用户
        self.statusBar().showMessage(u'已到达最后一组有标签变化的图像。')


    def openNextLabeledImg(self, _value=False):
        """
        跳转到下一张有标签的图像（考虑所有三个视图）。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 如果没有已打开的文件，尝试找到第一张有标签的文件
        if self.filePath is None:
            for i in range(len(self.mImgList)):
                if self.has_labels(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到有标签的图像。')
            return

        # 查找当前组索引
        current_group_index = next(
            (i for i, group in enumerate(self.mImgList) if self.filePath in group[1]),
            -1
        )

        # 从当前索引的下一个开始查找
        for i in range(current_group_index + 1, len(self.mImgList)):
            if self.has_labels(i):
                self.loadFile(i)
                return

        # 没有找到，提示用户
        self.statusBar().showMessage(u'已到达最后一张有标签的图像。')


    def openPrevChangedLabeledImg(self, _value=False):
        """
        跳转到上一组有标签但是标签有变化的图像（考虑所有三个视图）。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 如果没有已打开的文件，尝试找到最后一张有标签的文件
        if self.filePath is None:
            for i in reversed(range(len(self.mImgList))):
                if self.has_labels(i) and self.has_label_changes(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到标签变化的图像组。')
            return

        # 查找当前组索引
        current_group_index = next(
            (i for i, group in enumerate(self.mImgList) if self.filePath in group[1]),
            -1
        )

        # 从当前组索引的前一个开始查找
        for i in reversed(range(0, current_group_index)):
            if self.has_labels(i) and self.has_label_changes(i):
                self.loadFile(i)
                return

        # 没有找到，提示用户
        self.statusBar().showMessage(u'已到达第一组有标签变化的图像。')





    def openPrevLabeledImg(self, _value=False):
        """
        跳转到上一张有标签的图像（考虑所有三个视图）。
        """
        if self.autoSaving and self.defaultSaveDir and self.dirty:
            self.dirty = False
            self.canvas.verified = True
            self.saveFile()

        if not self.mayContinue():
            return

        if len(self.mImgList) == 0:
            return

        # 如果没有已打开的文件，尝试找到最后一张有标签的文件
        if self.filePath is None:
            for i in reversed(range(len(self.mImgList))):
                if self.has_labels(i):
                    self.loadFile(i)
                    return
            self.statusBar().showMessage(u'没有找到有标签的图像。')
            return

        # 查找当前组索引
        current_group_index = next(
            (i for i, group in enumerate(self.mImgList) if self.filePath in group[1]),
            -1
        )

        # 从当前索引的前一个开始查找
        for i in reversed(range(0, current_group_index)):
            if self.has_labels(i):
                self.loadFile(i)
                return

        # 没有找到，提示用户
        self.statusBar().showMessage(u'已到达第一张有标签的图像。')


    def has_labels(self, group_index):
        """
        判断指定组是否有标签（任何一个视图的图像有标签即可）。
        """
        prefix, image_paths = self.mImgList[group_index]

        for img_path in image_paths:
            annotation_filename = os.path.splitext(os.path.basename(img_path))[0] + XML_EXT
            annotation_path = os.path.join(self.defaultSaveDir, annotation_filename)
            if os.path.exists(annotation_path):  # 只要有一个文件存在，就返回True
                try:
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():  # 只要文件内容非空，就认为有标签
                                return True
                except:
                    pass  # 继续检查下一个文件

        return False  # 所有图像都没有标签

    def has_label_changes(self, group_index):
        """
        检查指定组的标签是否有变化（即图像组中不同视图的标签是否不同）。
        """
        prefix, image_paths = self.mImgList[group_index]
        labels = []

        for img_path in image_paths:
            annotation_filename = os.path.splitext(os.path.basename(img_path))[0] + XML_EXT
            annotation_path = os.path.join(self.defaultSaveDir, annotation_filename)
            if os.path.exists(annotation_path):
                try:
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        label_data = [line.strip() for line in f if line.strip()]
                        labels.append(label_data)
                except:
                    pass  # 继续检查下一个文件

        # 检查不同视图是否有不同的标签内容
        if len(labels) > 1:
            first_label_set = set(labels[0])
            for label in labels[1:]:
                if set(label) != first_label_set:
                    return True  # 标签有变化

        return False  # 标签没有变化


class Settings(object):
    """Convenience dict-like wrapper around QSettings."""

    def __init__(self, types=None):
        self.data = QSettings()
        self.types = defaultdict(lambda: QVariant, types if types else {})

    def __setitem__(self, key, value):
        t = self.types[key]
        self.data.setValue(key,
                           t(value) if not isinstance(value, t) else value)

    def __getitem__(self, key):
        return self._cast(key, self.data.value(key))

    def get(self, key, default=None):
        return self._cast(key, self.data.value(key, default))

    def _cast(self, key, value):
        # XXX: Very nasty way of converting types to QVariant methods :P
        t = self.types.get(key)
        if t is not None and t != QVariant:
            if t is str:
                return ustr(value)
            else:
                try:
                    method = getattr(QVariant, re.sub(
                        '^Q', 'to', t.__name__, count=1))
                    return method(value)
                except AttributeError as e:
                    # print(e)
                    return value
        return value


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    # Usage : labelImg.py image predefClassFile
    win = MainWindow(argv[1] if len(argv) >= 2 else None,
                     argv[2] if len(argv) >= 3 else os.path.join('data', 'predefined_classes.txt'))
    win.show()
    return app, win


def main(argv=[]):
    '''construct main app and run it'''
    app, _win = get_main_app(argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
