# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\imgViewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(35, 0))
        self.label.setMaximumSize(QtCore.QSize(16777215, 500))
        self.label.setScaledContents(False)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setMaximumSize(QtCore.QSize(16777215, 28))
        self.label_13.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout.addWidget(self.label_13)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setMaximumSize(QtCore.QSize(200, 16777215))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setMaximumSize(QtCore.QSize(16777215, 28))
        self.label_14.setObjectName("label_14")
        self.horizontalLayout.addWidget(self.label_14)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setMaximumSize(QtCore.QSize(16777215, 24))
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 3, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setMaximumSize(QtCore.QSize(16777215, 24))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 1, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setMaximumSize(QtCore.QSize(16777215, 24))
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 1, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 3, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 3, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setMaximumSize(QtCore.QSize(16777215, 24))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 3, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setMaximumSize(QtCore.QSize(200, 16777215))
        self.listView.setStyleSheet("QListView::item { border-bottom: 1px solid #ccc; }\n"
"QListView::item:selected { color: black; }")
        self.listView.setObjectName("listView")
        self.horizontalLayout_2.addWidget(self.listView)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuWindow = QtWidgets.QMenu(self.menubar)
        self.menuWindow.setObjectName("menuWindow")
        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionAnti_Alias = QtWidgets.QAction(MainWindow)
        self.actionAnti_Alias.setCheckable(True)
        self.actionAnti_Alias.setObjectName("actionAnti_Alias")
        self.actionOpen_Image_B = QtWidgets.QAction(MainWindow)
        self.actionOpen_Image_B.setObjectName("actionOpen_Image_B")
        self.actionHistogram_Normalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Normalization.setCheckable(True)
        self.actionHistogram_Normalization.setObjectName("actionHistogram_Normalization")
        self.actionHistogram = QtWidgets.QAction(MainWindow)
        self.actionHistogram.setObjectName("actionHistogram")
        self.actionNegative = QtWidgets.QAction(MainWindow)
        self.actionNegative.setObjectName("actionNegative")
        self.actionThresholding = QtWidgets.QAction(MainWindow)
        self.actionThresholding.setObjectName("actionThresholding")
        self.actionSlicing = QtWidgets.QAction(MainWindow)
        self.actionSlicing.setObjectName("actionSlicing")
        self.actionGray_Code_Slicing = QtWidgets.QAction(MainWindow)
        self.actionGray_Code_Slicing.setObjectName("actionGray_Code_Slicing")
        self.actionWater_Mark = QtWidgets.QAction(MainWindow)
        self.actionWater_Mark.setObjectName("actionWater_Mark")
        self.actionContrast_Stretching = QtWidgets.QAction(MainWindow)
        self.actionContrast_Stretching.setObjectName("actionContrast_Stretching")
        self.actionMedian = QtWidgets.QAction(MainWindow)
        self.actionMedian.setObjectName("actionMedian")
        self.actionPseudo_Median = QtWidgets.QAction(MainWindow)
        self.actionPseudo_Median.setObjectName("actionPseudo_Median")
        self.actionOutlier = QtWidgets.QAction(MainWindow)
        self.actionOutlier.setObjectName("actionOutlier")
        self.actionLowpass = QtWidgets.QAction(MainWindow)
        self.actionLowpass.setObjectName("actionLowpass")
        self.actionHighpass = QtWidgets.QAction(MainWindow)
        self.actionHighpass.setObjectName("actionHighpass")
        self.actionEdge_Crispening_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Crispening_1.setObjectName("actionEdge_Crispening_1")
        self.actionEdge_Crispening_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Crispening_2.setObjectName("actionEdge_Crispening_2")
        self.actionEdge_Crispening_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Crispening_3.setObjectName("actionEdge_Crispening_3")
        self.actionHigh_Boost = QtWidgets.QAction(MainWindow)
        self.actionHigh_Boost.setObjectName("actionHigh_Boost")
        self.actionRoberts_Operator = QtWidgets.QAction(MainWindow)
        self.actionRoberts_Operator.setObjectName("actionRoberts_Operator")
        self.actionSobel_Operator = QtWidgets.QAction(MainWindow)
        self.actionSobel_Operator.setObjectName("actionSobel_Operator")
        self.actionPrewitt_Operator = QtWidgets.QAction(MainWindow)
        self.actionPrewitt_Operator.setObjectName("actionPrewitt_Operator")
        self.actionHuffman = QtWidgets.QAction(MainWindow)
        self.actionHuffman.setObjectName("actionHuffman")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionOpen_Image_B)
        self.menuView.addAction(self.actionAnti_Alias)
        self.menuView.addAction(self.actionHistogram_Normalization)
        self.menuWindow.addAction(self.actionHistogram)
        self.menuWindow.addAction(self.actionNegative)
        self.menuWindow.addAction(self.actionThresholding)
        self.menuWindow.addAction(self.actionSlicing)
        self.menuWindow.addAction(self.actionGray_Code_Slicing)
        self.menuWindow.addAction(self.actionWater_Mark)
        self.menuWindow.addAction(self.actionContrast_Stretching)
        self.menuWindow.addAction(self.actionHuffman)
        self.menuFilter.addAction(self.actionOutlier)
        self.menuFilter.addAction(self.actionMedian)
        self.menuFilter.addAction(self.actionPseudo_Median)
        self.menuFilter.addAction(self.actionLowpass)
        self.menuFilter.addAction(self.actionHighpass)
        self.menuFilter.addAction(self.actionEdge_Crispening_1)
        self.menuFilter.addAction(self.actionEdge_Crispening_2)
        self.menuFilter.addAction(self.actionEdge_Crispening_3)
        self.menuFilter.addAction(self.actionHigh_Boost)
        self.menuFilter.addAction(self.actionRoberts_Operator)
        self.menuFilter.addAction(self.actionSobel_Operator)
        self.menuFilter.addAction(self.actionPrewitt_Operator)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuWindow.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.label_13.setText(_translate("MainWindow", "A"))
        self.label_14.setText(_translate("MainWindow", "B"))
        self.label_9.setText(_translate("MainWindow", "Saturation"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_6.setText(_translate("MainWindow", "TextLabel"))
        self.label_10.setText(_translate("MainWindow", "Hue"))
        self.label_11.setText(_translate("MainWindow", "Hue"))
        self.label_8.setText(_translate("MainWindow", "TextLabel"))
        self.label_7.setText(_translate("MainWindow", "TextLabel"))
        self.label_12.setText(_translate("MainWindow", "Brightness"))
        self.label_5.setText(_translate("MainWindow", "TextLabel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionAnti_Alias.setText(_translate("MainWindow", "Anti-Alias"))
        self.actionOpen_Image_B.setText(_translate("MainWindow", "Open Image B"))
        self.actionHistogram_Normalization.setText(_translate("MainWindow", "Histogram Normalization"))
        self.actionHistogram.setText(_translate("MainWindow", "Histogram"))
        self.actionNegative.setText(_translate("MainWindow", "Negative"))
        self.actionThresholding.setText(_translate("MainWindow", "Thresholding"))
        self.actionSlicing.setText(_translate("MainWindow", "Slicing"))
        self.actionGray_Code_Slicing.setText(_translate("MainWindow", "Gray Code Slicing"))
        self.actionWater_Mark.setText(_translate("MainWindow", "Water Mark"))
        self.actionContrast_Stretching.setText(_translate("MainWindow", "Contrast Stretching"))
        self.actionMedian.setText(_translate("MainWindow", "Median"))
        self.actionPseudo_Median.setText(_translate("MainWindow", "Pseudo Median"))
        self.actionOutlier.setText(_translate("MainWindow", "Outlier"))
        self.actionLowpass.setText(_translate("MainWindow", "Lowpass"))
        self.actionHighpass.setText(_translate("MainWindow", "Highpass"))
        self.actionEdge_Crispening_1.setText(_translate("MainWindow", "Edge Crispening 1"))
        self.actionEdge_Crispening_2.setText(_translate("MainWindow", "Edge Crispening 2"))
        self.actionEdge_Crispening_3.setText(_translate("MainWindow", "Edge Crispening 3"))
        self.actionHigh_Boost.setText(_translate("MainWindow", "High-Boost"))
        self.actionRoberts_Operator.setText(_translate("MainWindow", "Roberts Operator"))
        self.actionSobel_Operator.setText(_translate("MainWindow", "Sobel Operator"))
        self.actionPrewitt_Operator.setText(_translate("MainWindow", "Prewitt Operator"))
        self.actionHuffman.setText(_translate("MainWindow", "Huffman"))
