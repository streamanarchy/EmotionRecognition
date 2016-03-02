# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
from speech import speechUI
import sys
from features import voiceUI
from emotion import emotionUI
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


class mainDialog(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self,None)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.speechButton.clicked.connect(self.speechDialog)
        self.ui.voiceButton.clicked.connect(self.voiceDialog)
        self.ui.pushButton_2.clicked.connect(self.emotionDialog)

    def emotionDialog(self):
        self.emotionDialogUI = emotionUI()
        self.emotionDialogUI.show()
        self.emotionDialogUI.raise_()
        aF.recordAudioSegments('/home/project/Documents/Project/input/wav/',8,self.emotionDialogUI)

    def speechDialog(self):
        print "speechDialog"
        self.speechDialogUI = speechUI()
        self.speechDialogUI.show()
        self.speechDialogUI.raise_()
        aF.speechRecognition(self.speechDialogUI)


    def voiceDialog(self):
        self.voiceDialogUI = voiceUI()
        self.voiceDialogUI.show()
        self.voiceDialogUI.raise_()


if __name__ =="__main__":
    app = QtGui.QApplication(sys.argv)
    view = mainDialog()
    view.show()
    sys.exit(app.exec_())

