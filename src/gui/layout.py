# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'layout.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QSlider, QSpacerItem, QStatusBar,
    QTabWidget, QToolButton, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(652, 393)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 582))
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.label)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.NST_Tab = QWidget()
        self.NST_Tab.setObjectName(u"NST_Tab")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.NST_Tab.sizePolicy().hasHeightForWidth())
        self.NST_Tab.setSizePolicy(sizePolicy1)
        self.verticalLayout_7 = QVBoxLayout(self.NST_Tab)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")

        self.verticalLayout_8.addLayout(self.verticalLayout_4)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer_3)

        self.label_3 = QLabel(self.NST_Tab)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_8.addWidget(self.label_3)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.source_directory_edit = QLineEdit(self.NST_Tab)
        self.source_directory_edit.setObjectName(u"source_directory_edit")
        self.source_directory_edit.setEnabled(True)
        self.source_directory_edit.setReadOnly(True)

        self.horizontalLayout_5.addWidget(self.source_directory_edit)

        self.source_directory_button = QToolButton(self.NST_Tab)
        self.source_directory_button.setObjectName(u"source_directory_button")

        self.horizontalLayout_5.addWidget(self.source_directory_button)


        self.verticalLayout_8.addLayout(self.horizontalLayout_5)

        self.label_4 = QLabel(self.NST_Tab)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_8.addWidget(self.label_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.style_directory_edit = QLineEdit(self.NST_Tab)
        self.style_directory_edit.setObjectName(u"style_directory_edit")
        self.style_directory_edit.setReadOnly(True)

        self.horizontalLayout_6.addWidget(self.style_directory_edit)

        self.style_directory_button = QToolButton(self.NST_Tab)
        self.style_directory_button.setObjectName(u"style_directory_button")

        self.horizontalLayout_6.addWidget(self.style_directory_button)


        self.verticalLayout_8.addLayout(self.horizontalLayout_6)

        self.label_5 = QLabel(self.NST_Tab)
        self.label_5.setObjectName(u"label_5")

        self.verticalLayout_8.addWidget(self.label_5)

        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.output_directory_edit = QLineEdit(self.NST_Tab)
        self.output_directory_edit.setObjectName(u"output_directory_edit")
        self.output_directory_edit.setReadOnly(True)

        self.horizontalLayout_2.addWidget(self.output_directory_edit)

        self.edit_directory_button = QToolButton(self.NST_Tab)
        self.edit_directory_button.setObjectName(u"edit_directory_button")

        self.horizontalLayout_2.addWidget(self.edit_directory_button)


        self.verticalLayout_6.addLayout(self.horizontalLayout_2)


        self.verticalLayout_8.addLayout(self.verticalLayout_6)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.cc_label = QLabel(self.NST_Tab)
        self.cc_label.setObjectName(u"cc_label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.cc_label.sizePolicy().hasHeightForWidth())
        self.cc_label.setSizePolicy(sizePolicy2)
        self.cc_label.setMinimumSize(QSize(25, 0))
        self.cc_label.setMaximumSize(QSize(25, 16777215))
        self.cc_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.cc_label, 1, 2, 1, 1)

        self.gamma_label = QLabel(self.NST_Tab)
        self.gamma_label.setObjectName(u"gamma_label")
        sizePolicy2.setHeightForWidth(self.gamma_label.sizePolicy().hasHeightForWidth())
        self.gamma_label.setSizePolicy(sizePolicy2)
        self.gamma_label.setMinimumSize(QSize(25, 0))
        self.gamma_label.setMaximumSize(QSize(100, 16777215))
        self.gamma_label.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.gamma_label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.gridLayout.addWidget(self.gamma_label, 0, 2, 1, 1)

        self.cc_slider = QSlider(self.NST_Tab)
        self.cc_slider.setObjectName(u"cc_slider")
        self.cc_slider.setMaximum(100)
        self.cc_slider.setOrientation(Qt.Orientation.Horizontal)

        self.gridLayout.addWidget(self.cc_slider, 1, 1, 1, 1)

        self.label_7 = QLabel(self.NST_Tab)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout.addWidget(self.label_7, 0, 0, 1, 1)

        self.label_8 = QLabel(self.NST_Tab)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)

        self.gamma_slider = QSlider(self.NST_Tab)
        self.gamma_slider.setObjectName(u"gamma_slider")
        self.gamma_slider.setMaximum(16)
        self.gamma_slider.setSingleStep(1)
        self.gamma_slider.setOrientation(Qt.Orientation.Horizontal)
        self.gamma_slider.setInvertedAppearance(False)

        self.gridLayout.addWidget(self.gamma_slider, 0, 1, 1, 1)


        self.verticalLayout_8.addLayout(self.gridLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_8.addItem(self.verticalSpacer)

        self.pushButton = QPushButton(self.NST_Tab)
        self.pushButton.setObjectName(u"pushButton")

        self.verticalLayout_8.addWidget(self.pushButton)


        self.verticalLayout_7.addLayout(self.verticalLayout_8)

        self.tabWidget.addTab(self.NST_Tab, "")
        self.CycleGAN_Tab = QWidget()
        self.CycleGAN_Tab.setObjectName(u"CycleGAN_Tab")
        self.tabWidget.addTab(self.CycleGAN_Tab, "")

        self.horizontalLayout.addWidget(self.tabWidget)

        self.style_image_tab = QTabWidget(self.centralwidget)
        self.style_image_tab.setObjectName(u"style_image_tab")
        self.source_image_tab_2 = QWidget()
        self.source_image_tab_2.setObjectName(u"source_image_tab_2")
        self.style_image_tab.addTab(self.source_image_tab_2, "")
        self.output_image_tab_2 = QWidget()
        self.output_image_tab_2.setObjectName(u"output_image_tab_2")
        self.output_image_tab_2.setMinimumSize(QSize(25, 0))
        self.output_image_tab_2.setMaximumSize(QSize(25, 16777215))
        self.style_image_tab.addTab(self.output_image_tab_2, "")

        self.horizontalLayout.addWidget(self.style_image_tab)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.label_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.style_image_tab.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Project Name", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Source Image", None))
        self.source_directory_button.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Style Image", None))
        self.style_directory_button.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Output Directory", None))
        self.edit_directory_button.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.cc_label.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.gamma_label.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Gamma:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Color Control:", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.NST_Tab), QCoreApplication.translate("MainWindow", u"NST", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.CycleGAN_Tab), QCoreApplication.translate("MainWindow", u"CycleGAN", None))
        self.style_image_tab.setTabText(self.style_image_tab.indexOf(self.source_image_tab_2), QCoreApplication.translate("MainWindow", u"Source", None))
        self.style_image_tab.setTabText(self.style_image_tab.indexOf(self.output_image_tab_2), QCoreApplication.translate("MainWindow", u"Output", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"University of Illinois - CS 445 Computational Photography - Final Project - Matthew Poteshman, Paul Lambert - Fall 2024", None))
    # retranslateUi

