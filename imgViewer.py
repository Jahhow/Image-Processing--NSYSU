from imgViewerUi import Ui_MainWindow
from math import ceil, floor, sin, cos
import os
import numpy as np
import warnings
from struct import unpack
import sys
from typing import List
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

SPLASH_SIZE = QSize(500, 300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # splash screen
    flashPixmap = QPixmap(SPLASH_SIZE)
    flashPixmap.fill(Qt.GlobalColor.white)
    splash = QSplashScreen(pixmap=flashPixmap)
    splash.showMessage('<p style="font-size:60px;color:#666;">Image Viewer</h1><p style="font-size:28px;color:#aaa">涂家浩 M103040005</p>',
                       Qt.AlignmentFlag.AlignCenter, Qt.white)
    splash.show()


# WINDOW_SIZE=QSize(1024,768)
WINDOW_SIZE = QSize(800, 600)


def setFreeResize(qwidget: QWidget):
    # Allow resize smaller than content pixmap
    sizePolicy = qwidget.sizePolicy()
    sizePolicy.setVerticalPolicy(QSizePolicy.Ignored)
    sizePolicy.setHorizontalPolicy(QSizePolicy.Ignored)
    qwidget.setSizePolicy(sizePolicy)


def pseudoMedian(array: np.ndarray):
    indices = []
    maxmin = 0
    minmax = 255

    def indiceCombinations(l, r, count):  # [l,r)
        nonlocal array, maxmin, minmax
        if count <= 0:
            # print(indices)
            a = array[indices]
            _maxmin = a.min()
            _minmax = a.max()
            if _maxmin > maxmin:
                maxmin = _maxmin
            if _minmax < minmax:
                minmax = _minmax
            return
        for i in range(l, r-(count-1)):
            indices.append(i)
            indiceCombinations(i+1, r, count-1)
            indices.pop()
    indiceCombinations(0, len(array), (len(array)+1)//2)
    # print(type(maxmin)) # <class 'numpy.uint8'>
    # print(type(minmax)) # <class 'numpy.uint8'>
    median = (int(maxmin)+int(minmax))//2
    return median


class Histogram(QMainWindow):
    def __init__(self, parent, bits: np.ndarray):  # parent is needed for
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        super().__init__(parent)
        self.setWindowTitle("Histogram")
        self.resize(WINDOW_SIZE)
        gray = np.average(bits, axis=-1)

        fig = Figure(figsize=(5, 3))
        axes = fig.subplots(2, 2)
        figCanvas = FigureCanvasQTAgg(fig)

        axes[0, 0].set_xlim(-1, 256)
        axes[0, 0].hist(gray.flatten(), bins=256, color='gray')

        axes[0, 1].set_xlim(-1, 256)
        axes[0, 1].hist(bits[:, :, 0].flatten(), bins=256, color='r')

        axes[1, 0].set_xlim(-1, 256)
        axes[1, 0].hist(bits[:, :, 1].flatten(), bins=256, color='g')

        axes[1, 1].set_xlim(-1, 256)
        axes[1, 1].hist(bits[:, :, 2].flatten(), bins=256, color='b')

        self.setCentralWidget(figCanvas)
        self.show()


class Image:
    def __init__(self) -> None:
        self.__bits = None
        self.pixmap = None
        self.__grayBits = None

    @property
    def bits(self):
        return self.__bits

    @bits.setter
    def bits(self, bits):
        self.__bits = bits
        if bits is None:
            return
        self.__bits = bits
        self.__grayBits = None
        self.__grayBits_withWaterMark = None
        h, w, _ = bits.shape
        self.pixmap = QPixmap(
            QImage(bits.tobytes(), w, h, 3*w, QImage.Format.Format_RGB888))

    def getGrayBits(self):
        if self.__grayBits is None:
            if self.__bits is not None:
                gray: np.ndarray = np.average(self.__bits, axis=-1)
                # print(gray.dtype) # float64
                # gray = np.floor(gray) # not working for water mark
                gray = gray.astype(np.uint8)
                self.__grayBits = gray

        return self.__grayBits

    def getGrayBits_keepWaterMark(self):
        if self.__grayBits_withWaterMark is None:
            if self.__bits is not None:
                gray: np.ndarray = np.average(self.__bits, axis=-1)
                gray = gray.astype(np.uint8)
                gray = (gray & 254) | (self.__bits[..., 0] & 1)
                self.__grayBits_withWaterMark = gray

        return self.__grayBits_withWaterMark


class Thresholding(QWidget):
    def __init__(self, parent, grayBits: np.ndarray):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle('Thresholding')
        self.resize(WINDOW_SIZE)
        self.grayBits = grayBits
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.installEventFilter(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(256)
        self.slider.setValue(128)
        self.slider.valueChanged.connect(self.sliderValueChange)
        self.setThreshold(self.slider.value())
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.show()

    def sliderValueChange(self):
        self.setThreshold(self.slider.value())

    def setThreshold(self, thres):
        bits = np.where(self.grayBits >= thres, 255, 0)
        bits = bits.astype(np.uint8)
        h, w = self.grayBits.shape
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w, QImage.Format.Format_Grayscale8))
        self.setPixmapFit(self.label, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.label and event.type() == QEvent.Resize:
            self.setPixmapFit(self.label, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        return fitScaleRate


class MedianFilter(QWidget):
    def __init__(self, parent, bits: np.ndarray, usePseudoMedian=False):
        super().__init__(parent, Qt.Window)
        if usePseudoMedian:
            self.setWindowTitle('Median Filter')
        else:
            self.setWindowTitle('Median Filter')
        self.resize(WINDOW_SIZE)
        # self.progress=None
        self.usePseudoMedian = usePseudoMedian
        self.sliderValueChange_handled = False
        self.pixmap = None
        self.originalBits = bits
        # self.bits = np.zeros_like(self.originalBits)
        self.bits = self.originalBits.copy()
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.installEventFilter(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        setFreeResize(self.label)
        layout.addWidget(self.label)

        self.textLabel = QLabel()
        self.textLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.textLabel.setMaximumHeight(32)
        layout.addWidget(self.textLabel)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(19)
        self.slider.setValue(2 if usePseudoMedian else 6)
        self.slider.valueChanged.connect(self.sliderValueChange)
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.show()

        self.sliderValueChange()

    def sliderValueChange(self):
        # if self.progress is not None:
        #     self.progress.cancel()
        self.sliderValueChange_handled = False
        value = self.slider.value()
        if self.usePseudoMedian:
            self.textLabel.setText(f'{value}×{value} Pseudo Median Filter')
        else:
            self.textLabel.setText(f'{value}×{value} Median Filter')

        # bits = np.zeros_like(self.originalBits)
        bits = self.bits

        h, w, _ = self.originalBits.shape
        startI = -((value-1)//2)
        # progress = QProgressDialog(
        #     f'{value}×{value} Median Filter', 'Cancel', 0, 100, parent=self)
        # progress.setMinimumDuration(100)
        # self.progress=progress
        for y in range(h):
            # if y & 0b1 == 0:
            self.pixmap = QPixmap(QImage(bits.tobytes(),
                                         w, h, w*3, QImage.Format.Format_RGB888))
            self.setPixmapFit(self.label, self.pixmap)

            # progress.setValue(int(y*100/h))
            QApplication.processEvents()
            if self.sliderValueChange_handled:
                return
            if not self.isVisible():  # this window may have been closed. return, then.
                return
            # if self.progress is None:
            #     # already handled by later sliderValueChange
            #     return
            # if progress.wasCanceled():
            #     break

            for x in range(w):
                startX = x+startI
                startY = y+startI
                endX = startX+value
                endY = startY+value
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                array: np.ndarray = self.originalBits[startY:endY, startX:endX]
                array = array.reshape([-1, 3])
                if self.usePseudoMedian:
                    for i in range(3):
                        bits[y, x, i] = pseudoMedian(array[..., i])
                        # print('.',end='',flush=True)
                else:
                    array = np.median(array, axis=0)
                    # print(array.dtype) # float64
                    bits[y, x] = array
        # progress.setValue(100)
        self.sliderValueChange_handled = True
        # self.progress=None
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.label, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.label and event.type() == QEvent.Resize:
            if self.pixmap is not None:
                self.setPixmapFit(self.label, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        # fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        # return fitScaleRate


class ConvFilter(QWidget):
    Lowpass = 0
    Highpass = 1
    EdgeCrispening1 = 2
    EdgeCrispening2 = 3
    EdgeCrispening3 = 4
    HighBoost = 5


class ConvFilter(QWidget):
    Lowpass = 0
    Highpass = 1
    EdgeCrispening1 = 2
    EdgeCrispening2 = 3
    EdgeCrispening3 = 4
    HighBoost = 5

    class Kernel:
        def getLowpassKernel(size=3):
            return np.array([[1]*size]*size)

        def getHighpassKernel(size=3):
            k = np.array([[-1]*size]*size)
            center = size//2
            k[center, center] = size*size-1
            return k

        def getHighBoostKernel(size=3, A=1.1):
            k = np.array([[-1]*size]*size)
            center = size//2
            k[center, center] = size*size*A-1
            return k

    def __init__(self, parent, bits: np.ndarray, mode=ConvFilter.Lowpass):
        super().__init__(parent, Qt.Window)
        self.mode = mode
        if mode == ConvFilter.Lowpass:
            self.title = 'Lowpass Filter'
            self.setWindowTitle(self.title)
            self.kernel = ConvFilter.Kernel.getLowpassKernel
        elif mode == ConvFilter.Highpass:
            self.title = 'Highpass Filter'
            self.setWindowTitle(self.title)
            self.kernel = ConvFilter.Kernel.getHighpassKernel
        elif mode == ConvFilter.HighBoost:
            self.title = 'High-Boost Filter'
            self.setWindowTitle(self.title)
            self.kernel = ConvFilter.Kernel.getHighBoostKernel
        elif mode == ConvFilter.EdgeCrispening1:
            self.title = 'Edge Crispening 1'
            self.setWindowTitle(self.title)
            self.kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        elif mode == ConvFilter.EdgeCrispening2:
            self.title = 'Edge Crispening 2'
            self.setWindowTitle(self.title)
            self.kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        elif mode == ConvFilter.EdgeCrispening3:
            self.title = 'Edge Crispening 3'
            self.setWindowTitle(self.title)
            self.kernel = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

        self.resize(WINDOW_SIZE)
        self.pixmap = None
        self.originalBits = bits
        self.bits = self.originalBits.copy()
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.installEventFilter(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        setFreeResize(self.label)
        layout.addWidget(self.label)

        if not self.mode in [ConvFilter.EdgeCrispening1, ConvFilter.EdgeCrispening2, ConvFilter.EdgeCrispening3]:
            self.textLabel = QLabel()
            self.textLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.textLabel.setMaximumHeight(32)
            layout.addWidget(self.textLabel)

            if mode == ConvFilter.HighBoost:
                def buildSliderA():
                    layout = QHBoxLayout()

                    self.textLabelA = QLabel('A:')
                    self.textLabelA.setAlignment(Qt.AlignmentFlag.AlignRight)
                    self.textLabelA.setMaximumHeight(32)
                    layout.addWidget(self.textLabelA)

                    self.sliderA = QSlider(Qt.Orientation.Horizontal)
                    self.sliderA.setMinimum(10)  # 1.0
                    self.sliderA.setMaximum(30)  # 2.0
                    self.sliderA.setValue(22)
                    self.sliderA.valueChanged.connect(self.sliderValueChange)

                    layout.addWidget(self.sliderA)
                    return layout

                layout.addLayout(buildSliderA())

            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(1)
            self.slider.setMaximum(19)
            self.slider.setSingleStep(2)
            self.slider.setValue(3)
            self.slider.valueChanged.connect(self.sliderValueChange)
            layout.addWidget(self.slider)

            if mode == ConvFilter.HighBoost:
                self.slider.setValue(9)

        self.setLayout(layout)
        self.show()

        self.sliderValueChange()

    def sliderValueChange(self):
        from scipy import signal

        h, w, _ = self.originalBits.shape
        kernel = self.kernel
        if not self.mode in [ConvFilter.EdgeCrispening1, ConvFilter.EdgeCrispening2, ConvFilter.EdgeCrispening3]:
            value = self.slider.value()
            if self.mode == ConvFilter.HighBoost:
                A = self.sliderA.value()/10
                self.textLabel.setText(
                    f'{value}×{value} {self.title}.  A = {A}')
                kernel = kernel(value, A)
            else:
                self.textLabel.setText(f'{value}×{value} {self.title}')
                kernel = kernel(value)
        bits = []
        for i in range(3):
            a = signal.convolve2d(
                self.originalBits[..., i], kernel, boundary='symm', mode='same')
            if not self.mode in [ConvFilter.EdgeCrispening1, ConvFilter.EdgeCrispening2, ConvFilter.EdgeCrispening3]:
                a = a/(value*value)
            a = np.clip(a, 0, 255)
            bits.append(a.astype(np.uint8))
        bits = np.stack(bits, axis=-1)
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.label, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.label and event.type() == QEvent.Resize:
            if self.pixmap is not None:
                self.setPixmapFit(self.label, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        # fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        # return fitScaleRate


class EdgeDetection(QWidget):
    Roberts = 0
    Sobel = 1
    Prewitt = 2


class EdgeDetection(QWidget):
    Roberts = 0
    Sobel = 1
    Prewitt = 2

    class Kernel:
        def getRobertsOperators():
            return (
                np.array([[1, 0],
                          [0, -1]]),
                np.array([[0, -1],
                          [1, 0]])
            )

        def getSobelOperators():
            return (
                np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]),
                np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]]),
            )

        def getPrewittOperators():
            return (
                np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]]),
                np.array([[1, 1, 1],
                          [0, 0, 0],
                          [-1, -1, -1]]),
            )

    def __init__(self, parent, bits: np.ndarray, mode=EdgeDetection.Roberts):
        super().__init__(parent, Qt.Window)
        self.mode = mode
        if mode == EdgeDetection.Roberts:
            self.title = 'Roberts Operator'
            self.kernelGx, self.kernelGy = EdgeDetection.Kernel.getRobertsOperators()
        elif mode == EdgeDetection.Sobel:
            self.title = 'Sobel Operator'
            self.kernelGx, self.kernelGy = EdgeDetection.Kernel.getSobelOperators()
        elif mode == EdgeDetection.Prewitt:
            self.title = 'Prewitt Operator'
            self.kernelGx, self.kernelGy = EdgeDetection.Kernel.getPrewittOperators()
        self.setWindowTitle(self.title)

        self.resize(WINDOW_SIZE)
        self.pixmapGx = None
        self.pixmapGy = None
        self.pixmap = None
        self.originalBits = bits

        layout = QGridLayout()

        def addImageLabel(title, column):
            nonlocal layout
            imageLabel = QLabel()
            imageLabel.installEventFilter(self)
            imageLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            setFreeResize(imageLabel)
            layout.addWidget(imageLabel, 0, column)

            textLabel = QLabel(title)
            textLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            textLabel.setMaximumHeight(32)
            layout.addWidget(textLabel, 1, column)

            return imageLabel, textLabel

        self.imageLabelGx, self.textLabelGx = addImageLabel('Gx',   0)
        self.imageLabelGy, self.textLabelGy = addImageLabel('Gy',   1)
        self.imageLabel,   self.textLabel = addImageLabel('Both', 2)

        self.setLayout(layout)
        self.compute()
        self.show()

    def compute(self):
        from scipy import signal

        h, w, _ = self.originalBits.shape
        bitsGx = []
        bitsGy = []
        for i in range(3):
            aGx = signal.convolve2d(
                self.originalBits[..., i], self.kernelGx, boundary='symm', mode='same')
            # aGx = np.clip(aGx, 0, 255)
            bitsGx.append(aGx)

            aGy = signal.convolve2d(
                self.originalBits[..., i], self.kernelGy, boundary='symm', mode='same')
            # aGy = np.clip(aGy, 0, 255)
            bitsGy.append(aGy)

        bitsGx = np.stack(bitsGx, axis=-1)
        bitsGy = np.stack(bitsGy, axis=-1)
        bits: np.ndarray = (bitsGx*bitsGx+bitsGy*bitsGy)**.5
        bits = bits.astype(np.uint8)

        def tobytes(bitsG):
            bitsG = np.absolute(bitsG)
            bitsG = np.clip(bitsG, 0, 255)
            return bitsG.astype(np.uint8)
        self.pixmapGx = QPixmap(QImage(tobytes(bitsGx),
                                w, h, w*3, QImage.Format.Format_RGB888))
        self.pixmapGy = QPixmap(QImage(tobytes(bitsGy),
                                w, h, w*3, QImage.Format.Format_RGB888))
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.imageLabelGx, self.pixmapGx)
        self.setPixmapFit(self.imageLabelGy, self.pixmapGy)
        self.setPixmapFit(self.imageLabel, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.imageLabelGx and event.type() == QEvent.Resize:
            self.setPixmapFit(self.imageLabelGx, self.pixmapGx)
        if source is self.imageLabelGy and event.type() == QEvent.Resize:
            self.setPixmapFit(self.imageLabelGy, self.pixmapGy)
        if source is self.imageLabel and event.type() == QEvent.Resize:
            self.setPixmapFit(self.imageLabel, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        if pixmap is None:
            return
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        label.setPixmap(scaledPixmap)


class OutlierFilter(QWidget):
    def __init__(self, parent, bits: np.ndarray):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle('Outlier Filter')
        self.resize(WINDOW_SIZE)
        # self.progress=None
        self.sliderValueChange_handled = False
        self.pixmap = None
        self.originalBits = bits
        # self.bits = np.zeros_like(self.originalBits)
        self.bits = self.originalBits.copy()
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.installEventFilter(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        setFreeResize(self.label)
        layout.addWidget(self.label)

        self.textLabel = QLabel()
        self.textLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.textLabel.setMaximumHeight(32)
        layout.addWidget(self.textLabel)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(90)
        self.slider.valueChanged.connect(self.sliderValueChange)
        layout.addWidget(self.slider)

        self.setLayout(layout)
        self.show()

        self.sliderValueChange()

    def sliderValueChange(self):
        # if self.progress is not None:
        #     self.progress.cancel()
        self.sliderValueChange_handled = False
        thres = self.slider.value()
        kernelSize = 3
        self.textLabel.setText(f'Thres: {thres}')

        # bits = np.zeros_like(self.originalBits)
        bits = self.bits

        h, w, _ = self.originalBits.shape
        startI = -((kernelSize-1)//2)
        # progress = QProgressDialog(
        #     f'{value}×{value} Median Filter', 'Cancel', 0, 100, parent=self)
        # progress.setMinimumDuration(100)
        # self.progress=progress
        for y in range(h):
            self.pixmap = QPixmap(QImage(bits.tobytes(),
                                         w, h, w*3, QImage.Format.Format_RGB888))
            self.setPixmapFit(self.label, self.pixmap)

            # progress.setValue(int(y*100/h))
            QApplication.processEvents()
            if self.sliderValueChange_handled:
                return
            if not self.isVisible():  # this window may have been closed. return, then.
                return
            # if self.progress is None:
            #     # already handled by later sliderValueChange
            #     return
            # if progress.wasCanceled():
            #     break

            for x in range(w):
                startX = x+startI
                startY = y+startI
                endX = startX+kernelSize
                endY = startY+kernelSize
                centerXsub = -startI
                centerYsub = -startI
                if startX < 0:
                    centerXsub += startX
                    startX = 0
                if startY < 0:
                    centerYsub += startY
                    startY = 0
                endX = min(w, endX)
                endY = min(h, endY)
                array: np.ndarray = self.originalBits[startY:endY, startX:endX]
                subW = array.shape[1]
                centerI = subW*centerYsub+centerXsub
                array = array.reshape([-1, 3])
                center = array[centerI]
                neighbors = np.delete(array, centerI, axis=0)
                avg = np.average(neighbors, axis=0)
                # center-avg
                # np.abs(center-avg)
                center = np.where(np.abs(center-avg) > thres, avg, center)
                bits[y, x] = center
        # progress.setValue(100)
        self.sliderValueChange_handled = True
        # self.progress=None
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.label, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.label and event.type() == QEvent.Resize:
            if self.pixmap is not None:
                self.setPixmapFit(self.label, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        # fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        # return fitScaleRate


class ContrastStretching(QWidget):
    def __init__(self, parent, bits: np.ndarray):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle('Contrast Stretching')
        self.resize(WINDOW_SIZE)
        self.bits = bits
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.installEventFilter(self)
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.sliders = ()

        slider1 = QSlider(Qt.Orientation.Horizontal)
        slider1.setMinimum(0)
        slider1.setMaximum(255)
        slider1.setValue(40)
        slider1.valueChanged.connect(self.sliderValueChange)
        layout.addWidget(slider1)

        slider2 = QSlider(Qt.Orientation.Horizontal)
        slider2.setMinimum(0)
        slider2.setMaximum(255)
        slider2.setValue(255-40)
        slider2.valueChanged.connect(self.sliderValueChange)
        layout.addWidget(slider2)

        self.sliders = slider1, slider2

        self.setLayout(layout)
        self.sliderValueChange()
        self.show()

    def sliderValueChange(self):
        start, end = self.sliders[0].value(), self.sliders[1].value()
        if start == end:
            return
        if start > end:
            start, end = end, start
        bits = np.clip(self.bits, start, end)
        bits = bits.astype(np.float32)
        bits = (bits-start)*255/(end-start)
        bits = bits.astype(np.uint8)
        h, w, _ = self.bits.shape
        self.pixmap = QPixmap(QImage(bits.tobytes(),
                                     w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.label, self.pixmap)

    def eventFilter(self, source, event):
        if source is self.label and event.type() == QEvent.Resize:
            self.setPixmapFit(self.label, self.pixmap)
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        return fitScaleRate


class Slicing(QWidget):
    def __init__(self, parent, grayBits: np.ndarray, useGrayCode=False):
        super().__init__(parent, Qt.Window)
        if useGrayCode:
            self.setWindowTitle('Gray Code Slicing')
        else:
            self.setWindowTitle('Slicing')
        self.resize(WINDOW_SIZE)

        if useGrayCode:
            grayBits = np.bitwise_xor(grayBits, grayBits >> 1)

        h, w = grayBits.shape
        self.slicings = []
        self.pixmaps = []
        for i in range(7, -1, -1):
            slice = (grayBits >> i) & 1
            slice = np.where(slice != 0, 255, 0)
            slice = slice.astype(np.uint8)
            self.pixmaps.append(QPixmap(QImage(slice.tobytes(),
                                        w, h, w, QImage.Format.Format_Grayscale8)))
            self.slicings.append(slice)

        layout = QGridLayout()

        self.labels = []
        for row in range(2):
            for col in range(4):
                label = QLabel()
                label.installEventFilter(self)
                label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                layout.addWidget(label, row, col)
                self.labels.append(label)

        self.setLayout(layout)
        self.show()

    def eventFilter(self, source, event):
        try:
            i = self.labels.index(source)
            if event.type() == QEvent.Resize:
                self.setPixmapFit(source, self.pixmaps[i])
        except ValueError:
            pass
        return super().eventFilter(source, event)

    def setPixmapFit(self, label, pixmap):
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        return fitScaleRate


class Huffman(QWidget):
    class Node:
        def __init__(self, data=0, weight=0, left=None, right=None) -> None:
            self.data = data
            self.weight = weight
            self.left = left
            self.right = right
            self.huffmanCode = None

        def isLeaf(self):
            return self.left is None and self.right is None

        def __lt__(self, other):
            return self.weight < other.weight

        def fillEncodings(self, curCode=''):
            '''fillEncodings to leaf nodes'''
            if self.isLeaf():
                self.huffmanCode = curCode
                return

            if self.left is not None:
                self.left.fillEncodings(curCode=curCode+'0')
            if self.right is not None:
                self.right.fillEncodings(curCode=curCode+'1')

    def __init__(self, parent, grayBits: np.ndarray) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.resize(400, 600)
        self.setWindowTitle('Huffman')
        self.grayBits = grayBits

        layout = QVBoxLayout()
        listView = QListView()
        self.dataModel = QStringListModel()
        listView.setModel(self.dataModel)
        layout.addWidget(listView)
        self.setLayout(layout)
        self.show()
        self.buildHuffmanTree()

    def buildHuffmanTree(self):
        bins = np.zeros([256], dtype=np.uint32)
        for color in self.grayBits.flatten():
            bins[color] += 1
        nodes: List[Huffman.Node] = []
        for color, count in enumerate(bins):
            if count == 0:
                continue
            n = Huffman.Node(data=color, weight=count)
            nodes.append(n)
        leafNodes = nodes.copy()
        while len(nodes) >= 2:
            nodes.sort(reverse=True)
            left = nodes.pop()
            right = nodes.pop()
            n = Huffman.Node(weight=left.weight+right.weight,
                             left=left, right=right)
            nodes.append(n)

        nodes[0].fillEncodings()
        # display data
        data = [f'{n.data:3d}: {n.huffmanCode}' for n in leafNodes]
        self.dataModel.setStringList(data)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, bits=None, title='Image Viewer – 涂家浩 M103040005'):
        super().__init__()

        # uic.loadUi('imgViewer.ui', self)
        self.setupUi(self)

        self.resize(WINDOW_SIZE)
        self.setWindowTitle(title)

        self.image = Image()
        self.bitsA = self.bitsB = None
        self.bitsAscaled = self.bitsBscaled = None
        self.bitsBeforeNormalize = None

        # Actions
        self.actionOpen.triggered.connect(self.clickMenuOpenFile)
        self.actionOpen_Image_B.triggered.connect(
            lambda: self.clickMenuOpenFile(isB=True))
        self.antiAlias = self.actionAnti_Alias.isChecked()
        self.actionAnti_Alias.triggered.connect(self.setAntiAlias)
        self.actionHistogram_Normalization.triggered.connect(
            self.onActionHistogramNormalization)
        self.actionHistogram.triggered.connect(
            self.onActionHistogram)
        self.actionNegative.triggered.connect(
            self.on_actionNegative)
        self.actionThresholding.triggered.connect(
            self.on_actionThresholding)
        self.actionSlicing.triggered.connect(
            self.on_actionSlicing)
        self.actionGray_Code_Slicing.triggered.connect(
            self.on_actionGray_Code_Slicing)
        self.actionWater_Mark.triggered.connect(
            self.on_actionWater_Mark)
        self.actionContrast_Stretching.triggered.connect(
            self.on_actionContrast_Stretching)
        self.actionMedian.triggered.connect(
            self.on_actionMedian)
        self.actionPseudo_Median.triggered.connect(
            self.on_actionPseudo_Median)
        self.actionOutlier.triggered.connect(
            self.on_actionOutlier)
        self.actionLowpass.triggered.connect(
            self.on_actionLowpass)
        self.actionHighpass.triggered.connect(
            self.on_actionHighpass)
        self.actionEdge_Crispening_1.triggered.connect(
            self.on_actionEdge_Crispening_1)
        self.actionEdge_Crispening_2.triggered.connect(
            self.on_actionEdge_Crispening_2)
        self.actionEdge_Crispening_3.triggered.connect(
            self.on_actionEdge_Crispening_3)
        self.actionHigh_Boost.triggered.connect(
            self.on_actionHigh_Boost)
        self.actionRoberts_Operator.triggered.connect(
            self.on_actionRoberts_Operator)
        self.actionSobel_Operator.triggered.connect(
            self.on_actionSobel_Operator)
        self.actionPrewitt_Operator.triggered.connect(
            self.on_actionPrewitt_Operator)
        self.actionHuffman.triggered.connect(
            self.on_actionHuffman)

        self.listModel = QStringListModel(self.listView)
        self.listView.setModel(self.listModel)
        self.listView.setWordWrap(True)

        if bits is None:
            testImage = bytearray([
                255, 0, 0, 0, 255, 0, 0, 0, 255,
                225, 0, 0, 0, 225, 0, 0, 0, 225,
                200, 0, 0, 0, 200, 0, 0, 0, 200,
            ])
            self.setImageByteArray(testImage, 3, 3)
        else:
            self.setImage(bits)
            self.listView.hide()

        self.label.installEventFilter(self)
        self.label_2.installEventFilter(self)
        self.label_3.installEventFilter(self)
        self.label_4.installEventFilter(self)
        self.label_5.installEventFilter(self)
        self.label_6.installEventFilter(self)
        self.label_7.installEventFilter(self)
        self.label_8.installEventFilter(self)

        setFreeResize(self.label)
        setFreeResize(self.label_2)
        setFreeResize(self.label_3)
        setFreeResize(self.label_4)
        setFreeResize(self.label_5)
        setFreeResize(self.label_6)
        setFreeResize(self.label_7)
        setFreeResize(self.label_8)

        # setFocus to receive keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Slider
        self.horizontalSlider: QSlider
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(255)
        self.horizontalSlider.setValue(127)
        self.horizontalSlider.valueChanged.connect(self.sliderValueChange)
        self.showSlider(False)

        self.show()

    def showSlider(self, show):
        if show:
            self.horizontalSlider.show()
            self.label_13.show()
            self.label_14.show()
        else:
            self.horizontalSlider.hide()
            self.label_13.hide()
            self.label_14.hide()

    def HistogramNormalize(self, bits):
        if self.bitsBeforeNormalize is bits:
            return self.bitsNormalized

        self.bitsBeforeNormalize = bits

        bitsNormalized = bits.copy()
        h, w = self.bitsIntensity.shape
        histogram = np.zeros(256)
        for y in range(h):
            for x in range(w):
                histogram[self.bitsIntensity[y, x]] += 1
        averageCount = np.average(histogram)
        accumulate = 0
        map = np.zeros(256)
        startI = 0
        newI = 0
        for i in range(256):
            accumulate += histogram[i]
            while accumulate > averageCount:
                accumulate -= averageCount
                newI += 1
            if i == 0:
                map[i] = 1
            else:
                map[i] = (startI + newI)/2/i
            startI = newI

        for y in range(h):
            for x in range(w):
                bitsNormalized[y, x] = np.clip(
                    bitsNormalized[y, x] * map[self.bitsIntensity[y, x]], 0, 255).astype(np.uint8)

        self.bitsNormalized = bitsNormalized
        return bitsNormalized

    def onActionHistogram(self):
        Histogram(self, self.image.bits)

    def on_actionNegative(self):
        neg = 255-self.image.bits
        Main(neg, 'Negative')

    def on_actionThresholding(self):
        Thresholding(self, self.image.getGrayBits())

    def on_actionSlicing(self):
        Slicing(self, self.image.getGrayBits_keepWaterMark())

    def on_actionGray_Code_Slicing(self):
        Slicing(self, self.image.getGrayBits_keepWaterMark(), useGrayCode=True)

    def on_actionWater_Mark(self):
        if self.bitsA is None:
            self.clickMenuOpenFile()
            if self.bitsA is None:
                return
        if self.bitsB is None:
            self.clickMenuOpenFile(isB=True)
            if self.bitsB is None:
                return
        THRESHOLD = 128

        h, w, _ = self.bitsAscaled.shape
        hb, wb, _ = self.bitsBscaled.shape
        # leftTop index on A where B should be placed
        yb, xb = (h-hb)//2, (w-wb)//2
        # leftTop index on B where we would take
        croppedYb, croppedXb = max(-yb, 0), max(-xb, 0)
        hBcropped, wBcropped = min(hb, h), min(wb, w)
        bitB = self.bitsBscaled[croppedYb:croppedYb +
                                hBcropped, croppedXb:croppedXb+wBcropped]
        bitB = np.average(bitB, axis=-1).astype(np.uint8)
        bitB = np.where(bitB < THRESHOLD, 0, 1).astype(np.uint8)
        bitB = bitB[..., np.newaxis]
        # leftTop index on A where B-Cropped should be placed
        yb2, xb2 = max(0, yb), max(0, xb)
        bits = self.bitsAscaled.copy()
        # bits = np.average(bits, axis=-1).astype(np.uint8)
        bits[yb2:yb2+hBcropped, xb2:xb2+wBcropped] = (bits[yb2:yb2 +
                                                           hBcropped, xb2:xb2+wBcropped] & 254) | bitB
        # bits = bits[..., np.newaxis]
        # bits = np.repeat(bits, 3, axis=-1)
        Main(bits, title='Water Mark')

    def on_actionContrast_Stretching(self):
        ContrastStretching(self, self.image.bits)

    def on_actionMedian(self):
        MedianFilter(self, self.image.bits)

    def on_actionPseudo_Median(self):
        MedianFilter(self, self.image.bits, usePseudoMedian=True)

    def on_actionOutlier(self):
        OutlierFilter(self, self.image.bits)

    def on_actionLowpass(self):
        ConvFilter(self, self.image.bits)

    def on_actionHighpass(self):
        ConvFilter(self, self.image.bits, mode=ConvFilter.Highpass)

    def on_actionEdge_Crispening_1(self):
        ConvFilter(self, self.image.bits, mode=ConvFilter.EdgeCrispening1)

    def on_actionEdge_Crispening_2(self):
        ConvFilter(self, self.image.bits, mode=ConvFilter.EdgeCrispening2)

    def on_actionEdge_Crispening_3(self):
        ConvFilter(self, self.image.bits, mode=ConvFilter.EdgeCrispening3)

    def on_actionHigh_Boost(self):
        ConvFilter(self, self.image.bits, mode=ConvFilter.HighBoost)

    def on_actionRoberts_Operator(self):
        EdgeDetection(self, self.image.bits, mode=EdgeDetection.Roberts)

    def on_actionSobel_Operator(self):
        EdgeDetection(self, self.image.bits, mode=EdgeDetection.Sobel)

    def on_actionPrewitt_Operator(self):
        EdgeDetection(self, self.image.bits, mode=EdgeDetection.Prewitt)

    def on_actionHuffman(self):
        Huffman(self, self.image.getGrayBits())

    def onActionHistogramNormalization(self):
        if self.actionHistogram_Normalization.isChecked():
            bits = self.HistogramNormalize(self.image.bits)
            self.image.bits = bits
        else:
            bits = self.bitsBeforeNormalize
            self.image.bits = bits
        self.updateMain()

    def sliderValueChange(self):
        self.setABfraction(self.horizontalSlider.value())

    def toQImage(self, data, w, h, bytesPerLine=None, format=None):
        if bytesPerLine is None:
            bytesPerLine = w*3
        if format is None:
            format = QImage.Format.Format_RGB888

        qimage = QImage(data, w, h, bytesPerLine, format).convertToFormat(
            QImage.Format.Format_RGBX8888)
        return qimage

    def toNparray(self, qimage: QImage):
        w, h = qimage.width(), qimage.height()
        bits = qimage.bits().asarray(4*w*h)
        bits = np.array(bits, dtype=np.uint8)
        bits = bits.reshape([h, w, 4])
        bits = bits[..., :3]
        return bits

    # f range: [0,255]
    def setABfraction(self, f):
        if self.bitsAscaled is None or self.bitsBscaled is None:
            return
        ha, wa, _ = self.bitsAscaled.shape
        hb, wb, _ = self.bitsBscaled.shape
        w, h = max(wa, wb), max(ha, hb)
        bits = np.full([h, w, 3], 240, dtype=np.uint8)
        xa, ya = (w-wa)//2, (h-ha)//2
        xb, yb = (w-wb)//2, (h-hb)//2
        bits[ya:ya+ha, xa:xa+wa] = self.bitsAscaled
        # bits[yb:yb+hb,xb:xb+wb]=self.bitsBscaled
        bits[yb:yb+hb, xb:xb+wb] = (
            (bits[yb:yb+hb, xb:xb+wb].astype(np.uint)*(255-f) +
             self.bitsBscaled.astype(np.uint)*f)//255
        ).astype(np.uint8)

        if self.actionHistogram_Normalization.isChecked():
            bits = self.HistogramNormalize(bits)

        self.image.bits = bits
        self.updateMain()

    def updateMain(self):
        if self.mainfit:
            self.setMainFit()
        else:
            myScaledPixmap = self.transformImage(
                self.image.bits, self.fitScale*self.scale, self.radian, self.antiAlias)
            self.label.setPixmap(myScaledPixmap)

    def setImageByteArray(self, data: bytearray, w, h, bytesPerLine=None, format=None, isB=False):
        qimage = self.toQImage(data, w, h, bytesPerLine, format)

        # split R,G,B planes
        # use numpy to get faster
        bits = self.toNparray(qimage)
        self.setImage(bits, isB)

    def setImage(self, bits: np.ndarray, isB=False):
        h, w, _ = bits.shape
        if isB:
            self.bitsB = bits
            if self.bitsA is None:
                if self.actionHistogram_Normalization.isChecked():
                    bits = self.HistogramNormalize(bits)
                self.image.bits = bits
                self.setMainFit()
                return
        else:
            self.bitsA = bits
            if self.bitsB is None:
                if self.actionHistogram_Normalization.isChecked():
                    bits = self.HistogramNormalize(bits)
                self.image.bits = bits
                self.setMainFit()

        if self.bitsA is not None and self.bitsB is not None:
            # both bitsA bitsB are available.
            self.showSlider(True)
            self.bitsBscaled = self.bitsAscaled = None
            ha, wa, _ = self.bitsA.shape
            hb, wb, _ = self.bitsB.shape
            wha = ha+wa
            whb = hb+wb
            self.bitsAscaled = self.bitsA
            self.bitsBscaled = self.bitsB
            if wha > whb:
                self.bitsBscaled, qpixmap = self.transformImage(
                    self.bitsB, wha/whb, self.antiAlias, returnNparray=True)
            elif wha < whb:
                self.bitsAscaled, qpixmap = self.transformImage(
                    self.bitsA, whb/wha, self.antiAlias, returnNparray=True)
            self.setABfraction(self.horizontalSlider.value())
            # self.setMainFit()

        # ---------------------------------------------------------------------------------------------------
        # Setup RGB & HSL display

        def splitPixmapRGB(i):
            resultbits = np.zeros([h, w, 3], dtype=np.uint8)
            resultbits[:, :, i] = bits[:, :, i]
            return QPixmap(QImage(resultbits.tobytes(), w, h, 3*w, QImage.Format.Format_RGB888))

        self.pm2 = splitPixmapRGB(0)
        self.setPixmapFit(self.label_2, self.pm2)
        self.pm3 = splitPixmapRGB(1)
        self.setPixmapFit(self.label_3, self.pm3)
        self.pm4 = splitPixmapRGB(2)
        self.setPixmapFit(self.label_4, self.pm4)

        # HSL
        I = bits.sum(2)/3
        self.bitsIntensity = I.astype(np.uint8)
        self.pm5 = QPixmap(QImage(self.bitsIntensity.tobytes(),
                           w, h, w, QImage.Format.Format_Grayscale8))
        self.setPixmapFit(self.label_5, self.pm5)

        D = I
        # D = np.where(D == 0, 2**-24, D)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            S = (1-bits.min(2)/D)*255
        # print(S[:3, :3])
        self.pm6 = QPixmap(QImage(S.astype(np.uint8).tobytes(),
                           w, h, w, QImage.Format.Format_Grayscale8))
        self.setPixmapFit(self.label_6, self.pm6)

        R, G, B = bits[..., 0]/255, bits[..., 1]/255, bits[..., 2]/255
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            D = np.sqrt(np.square(R-G)+R+G-2*B)

            H = (2*R-G-B)/2/D
        H = np.arccos(H.clip(-1, 1))
        H = np.where(G < B, 2*np.pi-H, H)
        H = H*255/2/np.pi
        self.pm7 = QPixmap(QImage(H.astype(np.uint8).tobytes(),
                           w, h, w, QImage.Format.Format_Grayscale8))
        self.setPixmapFit(self.label_7, self.pm7)

        minColor = np.expand_dims(bits.min(2), -1)
        H2 = bits-minColor
        maxColor = np.expand_dims(H2.max(2), -1)
        H2 = H2*255.  # it overflows if don't use float
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            H2 = H2/maxColor
        self.pm8 = QPixmap(QImage(H2.astype(np.uint8).tobytes(),
                           w, h, w*3, QImage.Format.Format_RGB888))
        self.setPixmapFit(self.label_8, self.pm8)

    def clickMenuOpenFile(self, isB=False):
        dialogTitle = 'Open File'
        if isB:
            dialogTitle += ' B'
        fname, _ = QFileDialog.getOpenFileName(
            self, dialogTitle, filter='Images (*.pcx *.bmp)')
        if(fname == ''):
            return
        self.openFile(fname, isB)

    def openFile(self, fname, isB=False):
        # try:
        print('open file', fname)
        with open(fname, 'rb') as file:
            infoList = []
            B = file.read(1)[0]
            if(B == 10):  # PCX
                # decode header
                version = file.read(1)[0]
                d = {
                    0: '(0) PC Paintbrush version 2.5 using a fixed EGA palette',
                    2: '(2) PC Paintbrush version 2.8 using a modifiable EGA palette',
                    3: '(3) PC Paintbrush version 2.8 using no palette',
                    4: '(4) PC Paintbrush for Windows',
                    5: '(5) PC Paintbrush version 3.0, including 24-bit images'
                }
                infoList.append(d[version])

                RLE_enalbled = file.read(1)[0] == 1
                infoList.append(
                    'Run-length encoding (RLE)' if RLE_enalbled else 'No encoding (rarely used)')

                bitsPerPixel = file.read(1)[0]
                infoList.append(f'{bitsPerPixel} bits per pixel per plane')

                dimen = file.read(8)
                # to short int array little endian
                dimen = unpack('H'*(len(dimen)//2), dimen)
                infoList.append(f'Coords: {dimen}')

                w = dimen[2]-dimen[0]+1
                h = dimen[3]-dimen[1]+1
                infoList.append(f'Width: {w}, Height: {h}')

                dpi = file.read(4)
                # to short int array little endian
                dpi = unpack('H'*(len(dpi)//2), dpi)
                infoList.append(f'HDPI: {dpi[0]}, VDPI: {dpi[1]}')

                palette = file.read(48)
                # print('Header palette:',palette)
                file.read(1)  # reserved

                nPlane = file.read(1)[0]
                infoList.append(f'{nPlane} color plane(s)')

                shorts = file.read(8)
                # to short int array little endian
                shorts = unpack('H'*(len(shorts)//2), shorts)
                bytesPerLine = shorts[0]
                infoList.append(
                    f'{bytesPerLine} bytes per scan line per plane')
                ColorTableType = shorts[1]
                d = {
                    1: '(1) The palette contains monochrome or color information',
                    2: '(2) The palette contains grayscale information'
                }
                infoList.append(d[ColorTableType])
                infoList.append(
                    f'Source screen size is {shorts[2]}x{shorts[3]}')

                palette256 = None
                if bitsPerPixel == 8 and nPlane == 1 and ColorTableType == 1:  # should have palette256
                    file.seek(-768, os.SEEK_END)
                    # if file.read(1)[0]==12: # 1.pcx don't even have this mark, but has palette256 exist!
                    B = file.read(768)
                    palette256 = B
                infoList.append(
                    'No suffix 256 color palette' if palette256 is None else 'Using suffix 256 color palette')

                self.listModel.setStringList(infoList)

                # -----------------------------------------------------
                # decode image

                file.seek(128)

                # bytes=file.read(100)
                # print(bytes)
                # return

                # pxPerByte = 8//bitsPerPixel
                # initNshift = 8-bitsPerPixel
                # pxMask = (1<<bitsPerPixel) - 1
                # bytesPerPx = 1
                if palette256 is not None:
                    bytesPerLine *= 3
                image = bytearray(nPlane*bytesPerLine*h)
                # print(bytesPerLine)               # 274
                # print((w+pxPerByte-1)//pxPerByte) # 273
                # assert bytesPerLine == (w+pxPerByte-1)//pxPerByte
                for y in range(h):
                    for iPlane in range(nPlane):
                        istart = y*nPlane*bytesPerLine+iPlane
                        i = istart
                        while i < istart+nPlane*bytesPerLine:
                            B = file.read(1)[0]
                            if B & 0b1100_0000 == 0b1100_0000:
                                count = B & 0b111_111
                                B = file.read(1)[0]
                                for _ in range(count):
                                    if palette256 is not None:
                                        image[i:i+3] = palette256[B*3:B*3+3]
                                        i += 3
                                    else:
                                        image[i] = B
                                        i += nPlane
                            else:
                                if palette256 is not None:
                                    image[i:i+3] = palette256[B*3:B*3+3]
                                    i += 3
                                else:
                                    image[i] = B
                                    i += nPlane
                format = QImage.Format.Format_RGB888
                if nPlane == 1:
                    if bitsPerPixel == 1:
                        format = QImage.Format.Format_Mono
                    elif bitsPerPixel == 8:
                        if palette256 is None:
                            format = format = QImage.Format.Format_Grayscale8
                self.setImageByteArray(
                    image, w, h, bytesPerLine*nPlane, format, isB)

            elif B == ord('B'):
                # Bitmap – Windows 3.1x, 95, NT, ... etc.
                if file.read(1)[0] == ord('M'):
                    infoList.append('BMP – Windows 3.1x, 95, NT, ... etc.')

                    fileSize, _, startAddr, DIBsize, w, h, nPlane, bitsPerPixel = unpack(
                        '<IIIiIIHH', file.read(12+12+4))
                    infoList += [
                        f'File size: {fileSize}',
                        f'Start address: {startAddr}',
                        f'DIB Header Size: {DIBsize}',
                        f'Width: {w}',
                        f'Height: {h}',
                        f'Color Plane(s): {nPlane}',
                        f'bitsPerPixel: {bitsPerPixel}',
                    ]
                    compressMode, rawDataSize, hdpm, vdpm, nColorUsed, nImportantColor = unpack(
                        'IIiiII', file.read(4*6))
                    d = {
                        0: '(0) BI_RGB: 無壓縮',
                        1: '(1) BI_RLE8: RLE 8位元/像素',
                        2: '(2) BI_RLE4: RLE 4位元/像素',
                        3: '(3) BI_BITFIELDS: 位欄位或者霍夫曼1D壓縮（BITMAPCOREHEADER2）: 像素格式由位遮罩指定，或點陣圖經過霍夫曼1D壓縮（BITMAPCOREHEADER2）',
                        4: '(4) BI_JPEG: JPEG或RLE-24壓縮（BITMAPCOREHEADER2）: 點陣圖包含JPEG圖像或經過RLE-24壓縮（BITMAPCOREHEADER2）',
                        5: '(5) BI_PNG: PNG: 點陣圖包含PNG圖像',
                        6: '(6) BI_ALPHABITFIELDS: 位欄位: 針對Windows CE .NET 4.0及之後版本',
                    }
                    infoList += [
                        f'Compress Mode: {d[compressMode]}',
                        f'rawDataSize: {rawDataSize}',
                        f'hdpm: {hdpm}',
                        f'vdpm: {vdpm}',
                        'nColorUsed: ' +
                        (str(nColorUsed) if nColorUsed !=
                         0 else f'2^{bitsPerPixel}'),
                        f'nImportantColor: {nImportantColor}' if nImportantColor != 0 else 'All Color is important',
                    ]

                    self.listModel.setStringList(infoList)

                    # ----------------------------------------------------------
                    # decode image
                    file.seek(startAddr)
                    rowSize = (bitsPerPixel*w+31)//32*4
                    image = bytearray(rowSize*h)
                    for i in range(rowSize*(h-1), -1, -rowSize):
                        B = file.read(rowSize)
                        image[i:i+rowSize] = B
                    format = QImage.Format.Format_BGR888
                    if bitsPerPixel == 8:
                        format = QImage.Format.Format_Grayscale8
                    self.setImageByteArray(image, w, h, rowSize, format, isB)

        # except Exception as e:
        #     print(e)

    def setAntiAlias(self):
        self.antiAlias = self.actionAnti_Alias.isChecked()
        # print('setAntiAlias:', self.antiAlias)
        myScaledPixmap = self.transformImage(
            self.image.bits, self.fitScale*self.scale, self.radian, self.antiAlias)
        self.label.setPixmap(myScaledPixmap)

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_F:
                self.setMainFit()

    # trigger Scale & Rotate
    def wheelEvent(self, event):
        self.mainfit = False
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.AltModifier:  # ALT pressing
            dx = event.angleDelta().x()
            if dx > 0:
                self.radian += np.pi/12
            else:
                self.radian -= np.pi/12
        else:
            dy = event.angleDelta().y()
            if dy > 0:
                self.scale *= 1.25
            else:
                self.scale *= .8
        myScaledPixmap = self.transformImage(
            self.image.bits, self.fitScale*self.scale, self.radian, self.antiAlias)
        self.label.setPixmap(myScaledPixmap)

    def eventFilter(self, source, event):
        d = {
            self.label_2: self.pm2,
            self.label_3: self.pm3,
            self.label_4: self.pm4,
            self.label_5: self.pm5,
            self.label_6: self.pm6,
            self.label_7: self.pm7,
            self.label_8: self.pm8,
        }
        if (source in d.keys() and event.type() == QEvent.Resize):
            label = source
            # re-scale the pixmap when the label resizes
            self.setPixmapFit(label, d[label])
        # print('self.fit',self.mainfit)
        if self.mainfit and source is self.label and event.type() == QEvent.Resize:
            self.setMainFit()
        return super(QMainWindow, self).eventFilter(source, event)

    def rotate(self, x, y, radian):
        _sin = sin(radian)
        _cos = cos(radian)
        return _cos*x-_sin*y, _sin*x+_cos*y

    def transformImage(self, img: np.ndarray, scale, radian, antiAlias=False, returnNparray=False):
        w, h = img.shape[1], img.shape[0]
        midw, midh = w/2, h/2
        nw, nh = w*scale, h*scale
        x, y = self.rotate(nw, nh, radian)
        x2, y2 = self.rotate(nw, -nh, radian)
        rnw, rnh = max(abs(x), abs(x2)), max(abs(y), abs(y2))
        transX, transY = (rnw-nw)/2, (rnh-nh)/2
        rnw, rnh = int(rnw), int(rnh)

        # def getPixel(img, x, y, fill=240):
        #     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        #         return img[y, x]
        #     return [fill, fill, fill]

        # print('Original:', w, h)
        # print('Scaled:', rnw, rnh)
        resultImg = np.full([rnh, rnw, 3], 240, dtype=np.uint8)
        for y in range(rnh):
            for x in range(rnw):
                x2, y2 = x-transX, y-transY
                x2, y2 = x2/scale, y2/scale
                x2, y2 = x2-midw, y2-midh
                x2, y2 = self.rotate(x2, y2, -radian)
                x2, y2 = x2+midw, y2+midh

                if antiAlias:
                    if x2 < 0 or w < x2 or y2 < 0 or h < y2:
                        continue
                    x2, y2 = x2-.5, y2-.5
                    x2, y2 = np.clip(x2, 0, w-1), np.clip(y2, 0, h-1)
                    x1, y1 = floor(x2), floor(y2)
                    x3, y3 = ceil(x2), ceil(y2)
                    fx, fy = x2-x1, y2-y1
                    leftTop = img[y1, x1]
                    if fy > 0:
                        leftBottom = img[y3, x1]
                        left = leftTop*(1-fy)+leftBottom*fy
                    else:
                        left = leftTop

                    if fx > 0:
                        rightTop = img[y1, x3]
                        if fy > 0:
                            rightBottom = img[y3, x3]
                            right = rightTop*(1-fy)+rightBottom*fy
                        else:
                            right = rightTop
                        resultImg[y, x] = left*(1-fx)+right*fx
                    else:
                        resultImg[y, x] = left
                else:
                    x2, y2 = floor(x2), floor(y2)
                    if x2 < 0 or w <= x2 or y2 < 0 or h <= y2:
                        continue
                    resultImg[y, x] = img[y2, x2]

        qpixmap = QPixmap(QImage(resultImg.tobytes(), rnw,
                          rnh, 3*rnw, QImage.Format.Format_RGB888))
        if returnNparray:
            return resultImg, qpixmap
        return qpixmap

    def setMainFit(self):
        self.mainfit = True
        self.scale = 1.  # scale upon fitScale
        self.radian = 0
        self.fitScale = self.setPixmapFit(self.label, self.image.pixmap)

    def setPixmapFit(self, label, pixmap):
        # print(list(label.size()))
        scaledPixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio)
        # print(scaledPixmap.size())
        fitScaleRate = scaledPixmap.width()/pixmap.width()
        label.setPixmap(scaledPixmap)
        return fitScaleRate


if __name__ == '__main__':
    main = Main()
    splash.finish(main)

    countOpenFile = 0
    for argv in sys.argv[1:]:
        if argv == '-h':
            main.onActionHistogram()
        elif argv == '-t':
            main.on_actionThresholding()
        elif argv == '-m':
            main.on_actionMedian()
        elif argv == '-o':
            main.on_actionOutlier()
        elif argv == '-l':
            main.on_actionLowpass()
        elif argv == '-hf':
            main.on_actionHighpass()
        elif argv == '-hb':
            main.on_actionHigh_Boost()
        elif argv == '-huf':
            main.on_actionHuffman()
        else:
            if countOpenFile < 2:
                main.openFile(argv, isB=countOpenFile == 1)
                countOpenFile += 1

    app.exec_()
    sys.exit()
