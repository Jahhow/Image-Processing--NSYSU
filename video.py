from videoUi import Ui_MainWindow
from PIL import Image
from glob import glob
import numpy as np
from typing import List
import math
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

from matplotlib import pyplot as plt

SPLASH_SIZE = QSize(500, 300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # splash screen
    flashPixmap = QPixmap(SPLASH_SIZE)
    flashPixmap.fill(Qt.GlobalColor.white)
    splash = QSplashScreen(pixmap=flashPixmap)
    splash.showMessage('<p style="font-size:60px;color:#666;">Video</h1><p style="font-size:28px;color:#aaa">涂家浩 M103040005</p>',
                       Qt.AlignmentFlag.AlignCenter, Qt.white)
    splash.show()


DEBUG = False
# WINDOW_SIZE=QSize(1024,768)
WINDOW_SIZE = QSize(800, 600)
KERNEL_W = 8
KERNEL_H = 8
STRIDE = 1


def setFreeResize(qwidget: QWidget):
    # Allow resize smaller than content pixmap
    sizePolicy = qwidget.sizePolicy()
    sizePolicy.setVerticalPolicy(QSizePolicy.Policy.Ignored)
    sizePolicy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    qwidget.setSizePolicy(sizePolicy)


def setPixmapFit(label, pixmap):
    if pixmap is None:
        return
    scaledPixmap = pixmap.scaled(
        label.size(), Qt.AspectRatioMode.KeepAspectRatio)
    label.setPixmap(scaledPixmap)


class MImage:
    def __init__(self) -> None:
        self.pixmap = None
        self.npimage = None


VIDEO_SIZE = 256
SCALE = 1.5
GRAY = QColor('#777')
MY_SUB_BG = QColor('#FFF')


class VectorCanvas(QLabel):
    def __init__(self):
        super().__init__()
        size = int(VIDEO_SIZE*SCALE)
        self.setFixedSize(size, size)

        self.pen_color = GRAY
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.mpixmap = QPixmap(self.size())
        self.mpixmap.fill(Qt.GlobalColor.transparent)
        self.setPixmap(self.mpixmap)

    def clearCanvas(self):
        self.mpixmap.fill(MY_SUB_BG)
        self.setPixmap(self.mpixmap)

    # def resizeEvent(self, a0: QResizeEvent) -> None:
    #     print(self.size())
    #     return super().resizeEvent(a0)

    def drawArrow(self, startPoint: QPointF, endPoint: QPointF):
        startPoint *= SCALE
        endPoint *= SCALE
        dx, dy = endPoint.x() - startPoint.x(), endPoint.y() - startPoint.y()

        arrowLength = math.sqrt(dx ** 2 + dy ** 2)
        if arrowLength == 0:
            return
        normDx, normDy = dx / arrowLength, dy / arrowLength  # normalize

        # perpendicular vector
        perpX = -normDy
        perpY = normDx

        ARROW_SIZE = 4 * SCALE
        arrow_size = ARROW_SIZE*(1-math.exp(-arrowLength/ARROW_SIZE))

        leftX = endPoint.x() - arrow_size * normDx + arrow_size * perpX
        leftY = endPoint.y() - arrow_size * normDy + arrow_size * perpY
        leftPoint = QPointF(leftX, leftY)

        rightX = endPoint.x() - arrow_size * normDx - arrow_size * perpX
        rightY = endPoint.y() - arrow_size * normDy - arrow_size * perpY
        rightPoint = QPointF(rightX, rightY)

        painter = QPainter(self.mpixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        p = painter.pen()
        p.setWidth(1)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(startPoint, endPoint)
        painter.drawLine(leftPoint, endPoint)
        painter.drawLine(rightPoint, endPoint)
        # painter.end()
        self.setPixmap(self.mpixmap)


class ImageView(QLabel):
    def __init__(self, pixmap=None):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # setFreeResize(self)
        self.setPixmapFit(pixmap)

    def setPixmapFit(self, pixmap):
        self.mPixmap = pixmap
        if pixmap is None:
            return
        scaledPixmap = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.setPixmap(scaledPixmap)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        super().resizeEvent(a0)
        setPixmapFit(self, self.mPixmap)


class VideoEncoder(QMainWindow):
    class LossFun:
        def MAD(src, tar):
            a = tar.astype(np.int32)-src.astype(np.int32)
            a = np.abs(a)
            return np.sum(a)

        def MSE(src, tar):
            a = tar.astype(np.int32)-src.astype(np.int32)
            a = np.square(a)
            return a.sum()

        def PDC(src, tar):
            a = tar.astype(np.int32)-src.astype(np.int32)
            a = a > 20
            return a.sum()

        def IP(src, tar):
            a = tar.astype(np.int32)-src.astype(np.int32)
            b = np.abs(np.sum(a, axis=0)).sum()
            c = np.abs(np.sum(a, axis=1)).sum()
            return b+c

    def __init__(self, title='Video Encoder – 涂家浩 M103040005', lossFun='MAD'):
        super().__init__()
        self.resize(899, 871)
        self.setWindowTitle(title)

        fnames = glob('sequences/6.*')
        fnames.sort()
        if DEBUG:
            fnames = fnames[:4]

        mimages: List[MImage] = []
        for name in fnames:
            im = Image.open(name)
            mimage = MImage()
            mimage.pixmap = im.toqpixmap()
            mimage.npimage = np.array(im)
            mimages.append(mimage)
        self.mimages = mimages

        if lossFun == 'MAD':
            lossFun = VideoEncoder.LossFun.MAD
        self.lossFun = lossFun

        if lossFun is VideoEncoder.LossFun.MAD:
            self.lossFunName = 'MAD'
        elif lossFun is VideoEncoder.LossFun.MSE:
            self.lossFunName = 'MSE'
        elif lossFun is VideoEncoder.LossFun.PDC:
            self.lossFunName = 'PDC'
        elif lossFun is VideoEncoder.LossFun.IP:
            self.lossFunName = 'IP'

        verticalLayout = QVBoxLayout()

        #####################################################################

        self.textLabel = QLabel(f'1/{len(mimages)}')
        self.textLabel.setMaximumHeight(36)
        self.textLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        verticalLayout.addWidget(self.textLabel)

        ##################################################################

        gridLayout = QGridLayout()

        LABEL_SIZE = 300
        SUB_LABEL_SIZE = 80
        self.imgLabelSrc = ImageView(mimages[0].pixmap)
        self.imgLabelSrc.setMinimumSize(LABEL_SIZE, LABEL_SIZE)
        gridLayout.addWidget(self.imgLabelSrc, 0, 0)

        self.imgLabelSubSrc = ImageView()
        self.imgLabelSubSrc.setFixedSize(SUB_LABEL_SIZE, SUB_LABEL_SIZE)
        gridLayout.addWidget(self.imgLabelSubSrc, 0, 1,
                             Qt.AlignmentFlag.AlignCenter)

        srcLabel = QLabel('Source')
        # srcLabel.setFixedHeight(36)
        srcLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(srcLabel, 1, 0, 1, 2)

        self.imgLabelSubTar = ImageView()
        self.imgLabelSubTar.setFixedSize(SUB_LABEL_SIZE, SUB_LABEL_SIZE)
        gridLayout.addWidget(self.imgLabelSubTar, 0, 2,
                             Qt.AlignmentFlag.AlignCenter)

        self.imgLabelTar = ImageView(mimages[0].pixmap)
        self.imgLabelTar.setMinimumSize(LABEL_SIZE, LABEL_SIZE)
        gridLayout.addWidget(self.imgLabelTar, 0, 3)

        targetLabel = QLabel('Target')
        # targetLabel.setFixedHeight(36)
        targetLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gridLayout.addWidget(targetLabel, 1, 2, 1, 2)

        gridLayout.setColumnStretch(0, 1)
        gridLayout.setColumnStretch(3, 1)
        gridLayout.setRowStretch(0, 1)

        verticalLayout.addLayout(gridLayout, 1)

        ####################################################################

        rowLayout2 = QGridLayout()

        self.vectorCanvas = VectorCanvas()
        rowLayout2.addWidget(self.vectorCanvas, 0, 0)

        self.encodedPixmap = QPixmap(mimages[0].pixmap.size())
        self.encodedPixmap.fill(MY_SUB_BG)
        self.imgLabelEncoded = ImageView(self.encodedPixmap)
        # self.imgLabelEncoded.setMinimumSize(LABEL_SIZE, LABEL_SIZE)
        rowLayout2.addWidget(self.imgLabelEncoded, 0, 1)

        label = QLabel('Motion Vector')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rowLayout2.addWidget(label, 1, 0)
        label = QLabel('Encoded Result')
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rowLayout2.addWidget(label, 1, 1)

        verticalLayout.addLayout(rowLayout2)

        #####################################################################

        centralWidget = QWidget()
        centralWidget.setLayout(verticalLayout)
        self.setCentralWidget(centralWidget)

        # setFocus to receive keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def encode(self):
        canvas = self.vectorCanvas

        h, w = self.mimages[0].npimage.shape
        srcNpimage = self.mimages[0].npimage
        srcPixmap = self.mimages[0].pixmap
        encodedNpimage = np.zeros_like(self.mimages[0].npimage)
        vectorFrames = []
        for iFrame in range(1, len(self.mimages)):
            canvas.clearCanvas()
            self.textLabel.setText(
                f'{iFrame+1}/{len(self.mimages)}   LossFun: {self.lossFunName}')
            tarPixmap = self.mimages[iFrame].pixmap
            vectorRows = []
            for yTarget in range(0, h-KERNEL_H+1, KERNEL_H):
                aRowOfVectors = []
                for xTarget in range(0, w-KERNEL_W+1, KERNEL_W):
                    target = self.mimages[iFrame].npimage[yTarget:yTarget +
                                                          KERNEL_H, xTarget:xTarget+KERNEL_W]

                    minDiff = 0x7fff_ffff
                    for ySrc in range(0, h-KERNEL_H+1, STRIDE):
                        QApplication.processEvents()
                        # this window may have been closed. return, then.
                        if not self.isVisible():
                            sys.exit()
                            # self.close() # not working

                            # not working
                            # e = QKeyEvent(QEvent.KeyPress,Qt.Key_A,Qt.NoModifier,"a") # not working
                            # QApplication.postEvent(self, e) # not working
                            return
                        for xSrc in range(0, w-KERNEL_W+1, STRIDE):
                            src = srcNpimage[ySrc:ySrc +
                                             KERNEL_H, xSrc:xSrc+KERNEL_W]
                            a = self.lossFun(src, target)
                            if a < minDiff:
                                minDiff = a
                                resultX = xSrc
                                resultY = ySrc
                    aRowOfVectors.append([resultX, resultY])

                    # draw target rect
                    pixmap = tarPixmap.copy()
                    painter = QPainter(pixmap)
                    painter.setPen(QColorConstants.Cyan)
                    painter.drawRect(xTarget, yTarget, KERNEL_W, KERNEL_H)
                    painter.end()
                    self.imgLabelTar.setPixmapFit(pixmap)

                    # draw src rect
                    pixmap = srcPixmap.copy()
                    painter = QPainter(pixmap)
                    painter.setPen(QColorConstants.Magenta)
                    painter.drawRect(resultX, resultY, KERNEL_W, KERNEL_H)
                    painter.end()
                    self.imgLabelSrc.setPixmapFit(pixmap)

                    # draw encoded img
                    painter = QPainter(self.imgLabelEncoded.mPixmap)
                    painter.drawPixmap(QRectF(xTarget, yTarget, KERNEL_W, KERNEL_H),
                                       self.mimages[iFrame-1].pixmap, QRectF(resultX, resultY, KERNEL_W, KERNEL_H))
                    painter.end()
                    setPixmapFit(self.imgLabelEncoded,
                                 self.imgLabelEncoded.mPixmap)

                    encodedNpimage[yTarget:yTarget+KERNEL_H, xTarget:xTarget +
                                   KERNEL_W] = srcNpimage[resultY:resultY+KERNEL_H, resultX:resultX+KERNEL_W]

                    self.imgLabelSubTar.setPixmapFit(
                        tarPixmap.copy(xTarget, yTarget, KERNEL_W, KERNEL_H))
                    self.imgLabelSubSrc.setPixmapFit(
                        srcPixmap.copy(resultX, resultY, KERNEL_W, KERNEL_H))
                    canvas.drawArrow(QPointF(resultX+(KERNEL_W >> 1), resultY+(KERNEL_H >> 1)),
                                     QPointF(xTarget+(KERNEL_W >> 1), yTarget+(KERNEL_H >> 1)))
                vectorRows.append(aRowOfVectors)
            vectorFrames.append(vectorRows)
            srcNpimage = encodedNpimage.copy()
            srcPixmap = self.imgLabelEncoded.mPixmap.copy()
        npVectors = np.array(vectorFrames, dtype=np.uint16)
        np.save('Motion Vectors.npy', npVectors)

    # def keyPressEvent(self, event):
    #     if not event.isAutoRepeat():
    #         if event.key() == Qt.Key.Key_Space:
    #             self.onPlayButtonClick()


class TickThread(QThread):
    tick = pyqtSignal()

    def __init__(self):
        super().__init__()
        # self.stop = False

    def run(self, *args, **kwargs):
        while True:
            self.msleep(50)
            self.tick.emit()
            # if self.stop:
            #     return


class Mode:
    Play = 0
    Pause = 1


class VideoPlayer(QMainWindow, Ui_MainWindow):
    def __init__(self, title='Video – 涂家浩 M103040005', motionVectors=None):
        super().__init__()
        # uic.loadUi('video.ui', self)
        self.setupUi(self)
        self.resize(WINDOW_SIZE)
        self.setWindowTitle(title)
        self.setMode(Mode.Pause)
        self.oldMode = self.mode

        self.mimages: List[MImage] = []
        mimages = self.mimages
        fnames = glob('sequences/6.*')
        fnames.sort()
        for name in fnames:
            im = Image.open(name)
            mimage = MImage()
            mimage.pixmap = im.toqpixmap()
            mimage.npimage = np.array(im)
            mimages.append(mimage)

        self.videoLen = len(self.mimages)

        self.psnrs = []
        self.decodedMimages = None
        if motionVectors is not None:
            self.decodedMimages = []
            firstImg = Image.open('sequences/6.1.01.tiff')
            mimage = MImage()
            mimage.pixmap = firstImg.toqpixmap()
            mimage.npimage = np.array(firstImg)
            self.decodedMimages.append(mimage)

            if isinstance(motionVectors, str):
                motionVectors = np.load(motionVectors)
            srcPixmap = mimage.pixmap.copy()
            decodedPixmap = QPixmap(srcPixmap.size())
            for vectorFrame in motionVectors:
                painter = QPainter(decodedPixmap)
                yTarget = 0
                for vectorRow in vectorFrame:
                    xTarget = 0
                    for vector in vectorRow:
                        x, y = vector
                        painter.drawPixmap(QRectF(xTarget, yTarget, KERNEL_W, KERNEL_H),
                                           srcPixmap, QRectF(x, y, KERNEL_W, KERNEL_H))
                        xTarget += KERNEL_W
                    yTarget += KERNEL_H
                painter.end()

                mimage = MImage()
                mimage.pixmap = decodedPixmap
                mimage.npimage = decodedPixmap.toImage()

                def toNparray(qimage: QImage):
                    w, h = qimage.width(), qimage.height()
                    bits = qimage.bits().asarray(4*w*h)
                    bits = np.array(bits, dtype=np.uint8)
                    bits = bits.reshape([h, w, 4])
                    bits = bits[..., 0]
                    return bits
                mimage.npimage = toNparray(mimage.npimage)
                self.decodedMimages.append(mimage)

                srcPixmap = decodedPixmap
                decodedPixmap = QPixmap(srcPixmap.size())
            self.videoLen = min(self.videoLen, len(self.decodedMimages))

            for i in range(1, len(self.mimages)):
                mse = self.mimages[i].npimage-self.decodedMimages[i].npimage
                mse = mse**2
                mse = np.average(mse)
                psnr = 10*np.log10(255**2/mse)
                self.psnrs.append(psnr)

        self.imageLabel: QLabel
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.installEventFilter(self)
        setFreeResize(self.imageLabel)
        setPixmapFit(self.imageLabel, mimages[0].pixmap)

        self.decodedLabel: QLabel
        self.decodedLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.decodedLabel.installEventFilter(self)
        setFreeResize(self.decodedLabel)
        setPixmapFit(self.decodedLabel, mimages[0].pixmap)
        if motionVectors is None:
            self.decodedLabel.hide()

        self.timeLabel: QLabel
        self.timeLabel.setText(f'1/{self.videoLen}')

        # Actions
        self.actionEncoder.triggered.connect(
            self.on_actionEncoder)

        # Buttons
        self.playButton: QPushButton
        self.updatePlayButtonIcon()
        self.playButton.clicked.connect(self.onPlayButtonClick)

        setFreeResize(self.imageLabel)

        # setFocus to receive keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Slider
        self.slider: QSlider
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.videoLen-1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.sliderValueChange)
        self.slider.sliderPressed.connect(self.sliderPressed)
        self.slider.sliderReleased.connect(self.sliderReleased)

        # Thread --------------------------------------------------------
        self.tickThread = TickThread()
        self.tickThread.tick.connect(self.onTick)
        self.tickThread.start()

    def plotPSNR(self):
        if self.decodedMimages is not None:
            plt.ylim(0, 50)
            plt.plot(self.psnrs)
            plt.show()

    def sliderValueChange(self):
        self.timeLabel.setText(f'{self.slider.value()+1}/{self.videoLen}')
        self.updatePixmap()

    def sliderPressed(self):
        self.oldMode = self.mode
        self.mode = Mode.Pause
    # def closeEvent(self, a0: QCloseEvent) -> None:
    #     self.tickThread.stop=True
    #     return super().closeEvent(a0)

    def sliderReleased(self):
        self.mode = self.oldMode

    def updatePixmap(self):
        setPixmapFit(self.imageLabel,
                     self.mimages[self.slider.value()].pixmap)
        if self.decodedMimages is not None:
            setPixmapFit(self.decodedLabel,
                         self.decodedMimages[self.slider.value()].pixmap)

    def onTick(self):
        if self.mode == Mode.Play:
            if self.slider.value() < self.videoLen-1:
                newPos = self.slider.value()+1
            else:
                newPos = 0
            self.slider.setValue(newPos)
            self.sliderValueChange()
    # def closeEvent(self,event):
    #     self.tickThread.stop=True
    #     return super().closeEvent(event)

    def updatePlayButtonIcon(self):
        if self.mode == Mode.Play:
            self.playButton.setIcon(self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPause))
        elif self.mode == Mode.Pause:
            self.playButton.setIcon(self.style().standardIcon(
                QStyle.StandardPixmap.SP_MediaPlay))

    def onPlayButtonClick(self):
        if self.mode == Mode.Play:
            self.mode = Mode.Pause
            self.updatePlayButtonIcon()
        elif self.mode == Mode.Pause:
            self.mode = Mode.Play
            self.updatePlayButtonIcon()

    def on_actionEncoder(self):
        self.setMode(Mode.Pause)
        self.videoEncoder = VideoEncoder()
        self.videoEncoder.show()
        self.videoEncoder.encode()

    def setMode(self, mode):
        self.mode = mode
        self.updatePlayButtonIcon()

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            if event.key() == Qt.Key.Key_Space:
                self.onPlayButtonClick()

    def eventFilter(self, source, event):
        if source in [self.imageLabel, self.decodedLabel] and event.type() == QEvent.Type.Resize:
            self.updatePixmap()
        return super(QMainWindow, self).eventFilter(source, event)


if __name__ == '__main__':
    window = VideoPlayer
    motionVectorName = None
    lossFun = VideoEncoder.LossFun.MAD
    for argv in sys.argv[1:]:
        if argv == '-debug':
            DEBUG = True
            STRIDE = 32
        elif argv == '-e':
            window = VideoEncoder
        elif argv == '-MSE' or argv == '-mse':
            lossFun = VideoEncoder.LossFun.MSE
        elif argv == '-PDC' or argv == '-pdc':
            lossFun = VideoEncoder.LossFun.PDC
        elif argv == '-IP' or argv == '-ip':
            lossFun = VideoEncoder.LossFun.IP
        else:
            motionVectorName = argv

    if window is VideoPlayer:
        window = window(motionVectors=motionVectorName)
        splash.finish(window)
        window.show()
        window.plotPSNR()
    else:
        window = window(lossFun=lossFun)
        splash.finish(window)
        window.show()
        window.encode()

    app.exec()
