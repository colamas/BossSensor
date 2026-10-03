# -*- coding: utf-8 -*-
import sys

from PyQt6 import QtCore, QtGui, QtWidgets


class _FullScreenImage(QtWidgets.QLabel):
    """Full screen image that closes on any key press or mouse click."""

    def keyPressEvent(self, event):
        self.close()

    def mousePressEvent(self, event):
        self.close()


def show_image(image_path='s_pycharm.jpg'):
    """Show the image full screen and block until it is closed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    screen = _FullScreenImage()
    screen.setPixmap(QtGui.QPixmap(image_path))
    screen.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    screen.setStyleSheet('background-color: black;')
    screen.showFullScreen()
    app.exec()


if __name__ == '__main__':
    show_image()
