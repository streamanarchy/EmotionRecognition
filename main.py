# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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
        self.speechButton = QtGui.QPushButton(Dialog)
        self.speechButton.setGeometry(QtCore.QRect(10, 40, 161, 41))
        self.speechButton.setObjectName(_fromUtf8("speechButton"))
        self.voiceButton = QtGui.QPushButton(Dialog)
        self.voiceButton.setGeometry(QtCore.QRect(210, 40, 161, 41))
        self.voiceButton.setObjectName(_fromUtf8("voiceButton"))
        self.pushButton_2 = QtGui.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(110, 150, 161, 41))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.speechButton.setText(_translate("Dialog", "Speech Recognition", None))
        self.voiceButton.setText(_translate("Dialog", "Voice Features", None))
        self.pushButton_2.setText(_translate("Dialog", "Emotion Recognition", None))

