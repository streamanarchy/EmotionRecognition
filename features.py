# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'features.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import allFunction as aF

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(400, 300)
        self.closeButton = QtGui.QPushButton(Dialog)
        self.closeButton.setGeometry(QtCore.QRect(300, 260, 88, 27))
        self.closeButton.setObjectName(_fromUtf8("closeButton"))
        self.zcrButton = QtGui.QPushButton(Dialog)
        self.zcrButton.setGeometry(QtCore.QRect(30, 20, 88, 27))
        self.zcrButton.setObjectName(_fromUtf8("zcrButton"))
        self.energyButton = QtGui.QPushButton(Dialog)
        self.energyButton.setGeometry(QtCore.QRect(170, 20, 88, 27))
        self.energyButton.setObjectName(_fromUtf8("energyButton"))
        self.energyEtButton = QtGui.QPushButton(Dialog)
        self.energyEtButton.setGeometry(QtCore.QRect(30, 60, 88, 27))
        self.energyEtButton.setObjectName(_fromUtf8("energyEtButton"))
        self.centroidButton = QtGui.QPushButton(Dialog)
        self.centroidButton.setGeometry(QtCore.QRect(170, 60, 88, 27))
        self.centroidButton.setObjectName(_fromUtf8("centroidButton"))
        self.fluxButton = QtGui.QPushButton(Dialog)
        self.fluxButton.setGeometry(QtCore.QRect(170, 100, 88, 27))
        self.fluxButton.setObjectName(_fromUtf8("fluxButton"))
        self.spreadButton = QtGui.QPushButton(Dialog)
        self.spreadButton.setGeometry(QtCore.QRect(30, 100, 88, 27))
        self.spreadButton.setObjectName(_fromUtf8("spreadButton"))
        self.segButton = QtGui.QPushButton(Dialog)
        self.segButton.setGeometry(QtCore.QRect(170, 140, 111, 31))
        self.segButton.setObjectName(_fromUtf8("segButton"))
        self.rollButton = QtGui.QPushButton(Dialog)
        self.rollButton.setGeometry(QtCore.QRect(30, 140, 88, 27))
        self.rollButton.setObjectName(_fromUtf8("rollButton"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.closeButton.setText(_translate("Dialog", "Close", None))
        self.zcrButton.setText(_translate("Dialog", "ZCR", None))
        self.energyButton.setText(_translate("Dialog", "Energy", None))
        self.energyEtButton.setText(_translate("Dialog", "Energy Et", None))
        self.centroidButton.setText(_translate("Dialog", "Centroid", None))
        self.fluxButton.setText(_translate("Dialog", "Flux", None))
        self.spreadButton.setText(_translate("Dialog", "Spread", None))
        self.segButton.setText(_translate("Dialog", "Segmentation", None))
        self.rollButton.setText(_translate("Dialog", "Rolloff", None))


class voiceUI(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QWidget.__init__(self,None)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.closeButton.clicked.connect(self.closeButton)
        """self.ui.zcrButton.clicked.connect(self.zcrButton)
        self.ui.energyButton.clicked.connect(self.energyButton)
        self.ui.energyEtButton.clicked.connect(self.energyEtButton)
        self.ui.centroidButton.clicked.connect(self.centroidButton)
        self.ui.fluxButton.clicked.connect(self.fluxButton)
        self.ui.spreadButton.clicked.connect(self.spreadButton)
        self.ui.rollButton.clicked.connect(self.rollButton)"""
        self.ui.segButton.clicked.connect(self.segButton)

    def closeButton(self):
        self.close()

    def segButton(self):
        aF.recordAudioSegments('/home/project/Documents/Project/input/wav/',8,'seg')