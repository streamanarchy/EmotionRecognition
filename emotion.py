# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'emotion.ui'
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
        Dialog.resize(781, 474)
        self.speechText = QtGui.QTextBrowser(Dialog)
        self.speechText.setGeometry(QtCore.QRect(40, 50, 256, 192))
        self.speechText.setObjectName(_fromUtf8("speechText"))
        self.featureText = QtGui.QTextBrowser(Dialog)
        self.featureText.setGeometry(QtCore.QRect(440, 50, 256, 192))
        self.featureText.setObjectName(_fromUtf8("featureText"))
        self.Close = QtGui.QPushButton(Dialog)
        self.Close.setGeometry(QtCore.QRect(670, 420, 88, 27))
        self.Close.setObjectName(_fromUtf8("Close"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.Close.setText(_translate("Dialog", "Close", None))


class emotionUI(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QWidget.__init__(self,None)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.Close.clicked.connect(self.closeButton)

    def closeButton(self):
        self.close()