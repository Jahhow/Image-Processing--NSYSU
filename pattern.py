from typing import List
import matplotlib.pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image

SPLASH_SIZE = QSize(500, 300)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # splash screen
    flashPixmap = QPixmap(SPLASH_SIZE)
    flashPixmap.fill(Qt.GlobalColor.white)
    splash = QSplashScreen(pixmap=flashPixmap)
    splash.showMessage('<p style="font-size:60px;color:#666;">Pattern</h1><p style="font-size:28px;color:#aaa">涂家浩 M103040005</p>',
                       Qt.AlignmentFlag.AlignCenter, Qt.white)
    splash.show()

DEBUG = False

# WINDOW_SIZE=QSize(1024,768)
WINDOW_SIZE = QSize(800, 600)

GRAY = QColor('#777')
RED = QColor('#f54242')
LIGHT_GREEN = QColor('#c0eb34')
GREEN = QColor('#00cf0a')
BLUE = QColor('#007bff')
PURPLE = QColor('#b434eb')


def setFreeResize(qwidget: QWidget):
    # Allow resize smaller than content pixmap
    sizePolicy = qwidget.sizePolicy()
    sizePolicy.setVerticalPolicy(QSizePolicy.Ignored)
    sizePolicy.setHorizontalPolicy(QSizePolicy.Ignored)
    qwidget.setSizePolicy(sizePolicy)


def setPixmapFit(label, pixmap):
    if pixmap is None:
        return
    scaledPixmap = pixmap.scaled(
        label.size(), Qt.AspectRatioMode.KeepAspectRatio)
    label.setPixmap(scaledPixmap)


class MImage:
    def __init__(self):
        self.pixmap = None
        self.npimg = None


class ImageView(QLabel):
    def __init__(self, pixmap=None):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        setFreeResize(self)
        self.setPixmapFit(pixmap)

    def setPixmapFit(self, pixmap):
        self.mPixmap = pixmap
        setPixmapFit(self, pixmap)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        super().resizeEvent(a0)
        setPixmapFit(self, self.mPixmap)


def vlen(v: QPointF):
    return (v.x()**2+v.y()**2)**.5


def dist(a: QPointF, b: QPointF):
    v = b-a
    return vlen(v)


def crossFun(a: QPointF, b: QPointF):
    return a.x()*b.y()-b.x()*a.y()


def ccw(a: QPointF, b: QPointF, c: QPointF):
    return crossFun(b-a, c-b)


def normalize(p: QPointF):
    l = vlen(p)
    p /= l


class Pattern(QWidget):
    Roberts = 0
    Sobel = 1
    Prewitt = 2


class MyPath:
    def __init__(self) -> None:
        self.points = None
        self.crosses = None


class Pattern(QWidget):
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

    class WorkThread(QThread):
        setPixmapFit = pyqtSignal(ImageView, QPixmap)
        plot = pyqtSignal(list)
        text = pyqtSignal(str)

        def __init__(self, pattern: Pattern):
            super().__init__(pattern)
            self.pattern = pattern

        def run(self):
            self.paintDetectedEdges()
            self.determineSizePos()
            return

        def paintDetectedEdges(self):
            global DEBUG
            self.circles: List[MyPath] = []
            self.rects: List[MyPath] = []
            h, w = self.pattern.originImg.npimg.shape
            self.pattern.detectedEdgesPixmap = self.pattern.convdImg.pixmap.copy()
            pixmap = self.pattern.detectedEdgesPixmap

            PEN_RADIUS = 15
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(PURPLE))

            thres = 50
            step = 5
            closePointThres = step
            crossThres = .1

            npimg = self.pattern.convdImg.npimg
            globalPointHistory = []
            for y in range(h):
                for x in range(w):
                    if npimg[y, x] > thres:
                        p = QPointF(x, y)

                        continue2 = False
                        for oldP in globalPointHistory:
                            if dist(oldP, p) < step*2:
                                continue2 = True
                        if continue2:
                            continue

                        points = [p]
                        direction = None
                        crosses = []
                        continueNextStartPoint = False

                        def f():
                            nonlocal direction, points, continueNextStartPoint
                            while True:
                                def neighborEdgePoints(center: QPointF):
                                    x, y = int(center.x()), int(center.y())
                                    y2 = y-step
                                    if 0 <= y2:
                                        for x2 in range(max(0, x-step), min(w, x+step+1)):
                                            if npimg[y2, x2] > thres:
                                                yield QPointF(x2, y2)
                                    x2 = x-step
                                    if 0 <= x2:
                                        for y2 in range(max(0, y-step+1), min(h, y+step)):
                                            if npimg[y2, x2] > thres:
                                                yield QPointF(x2, y2)
                                    x2 = x+step
                                    if x2 < w:
                                        for y2 in range(max(0, y-step+1), min(h, y+step)):
                                            if npimg[y2, x2] > thres:
                                                yield QPointF(x2, y2)
                                    y2 = y+step
                                    if y2 < h:
                                        for x2 in range(max(0, x-step), min(w, x+step+1)):
                                            if npimg[y2, x2] > thres:
                                                yield QPointF(x2, y2)
                                lastPoint = points[len(points)-1]
                                lastPoint2 = None if len(
                                    points)-2 < 0 else points[len(points)-2]
                                for p in neighborEdgePoints(lastPoint):
                                    newDirection = p-lastPoint
                                    normalize(newDirection)

                                    if direction is None:
                                        direction = newDirection
                                        points.append(p)
                                        break

                                    curCross = crossFun(
                                        direction, newDirection)
                                    if lastPoint2 is None:
                                        direction = newDirection
                                        points.append(p)
                                        break

                                    if dist(points[0], p) < closePointThres:
                                        # print(dist(newDirection,direction))
                                        return
                                    if len(points) >= 2:
                                        if dist(points[1], p) < closePointThres:
                                            # print(dist(newDirection,direction))
                                            return
                                    if dist(lastPoint2, p) < closePointThres:
                                        # print(dist(newDirection,direction))
                                        continue

                                    continue2 = False
                                    for oldP in points:
                                        dist(oldP, p)
                                        if dist(oldP, p) < closePointThres:
                                            continue2 = True
                                            break
                                    if continue2:
                                        continue

                                    # print(f'p: {p}, cross: {curCross}')
                                    points.append(p)
                                    curCross = crossFun(
                                        direction, newDirection)
                                    crosses.append(curCross)
                                    direction = newDirection
                                    break
                                else:
                                    # print('No neighbor')
                                    continueNextStartPoint = True
                                    return
                        f()
                        if continueNextStartPoint:
                            continue
                        if len(points) <= 6:
                            continue

                        sortCrosses = np.array(crosses)
                        sortCrosses.sort()
                        iLast = len(sortCrosses)-1
                        variance = np.var(sortCrosses)
                        median = sortCrosses[int(iLast*.5)]
                        median1_4 = sortCrosses[int(iLast*.25)]
                        median3_4 = sortCrosses[int(iLast*.75)]
                        avg = np.average(crosses)
                        _min = np.min(sortCrosses[0])
                        _max = np.max(sortCrosses[iLast])
                        _range = _max-_min

                        globalPointHistory.extend(points)

                        shape = None

                        def isRect():
                            return median == 0 and median1_4 == 0 and median3_4 == 0 and _range < 1.5

                        def isCircle():
                            return variance < .035

                        mPath = MyPath()
                        mPath.points = points
                        mPath.crosses = crosses
                        if isRect():
                            shape = 'Rect'
                            self.rects.append(mPath)
                        elif isCircle():
                            shape = 'Circle'
                            self.circles.append(mPath)
                        else:
                            continue

                        print(
                            '\n==========================================================')
                        # for i in range(len(points)):
                        #     p = points[i]
                        #     s = f'i: {(i+1)/len(points):.2f},  P: ({p.x():3.0f}, {p.y():3.0f})'
                        #     if i >= 2:
                        #         s += f',  Cross: {crosses[i-2]:5.2f},  SortCross: {sortCrosses[i-2]:5.2f}'
                        #     print(s)
                        print('variance:', variance)
                        print('median 1/4:', median1_4)
                        print('median:', median)
                        print('median 3/4:', median3_4)
                        print('avg:', avg)
                        print('range:', _range)
                        print('Found', shape)

                        painter.drawEllipse(points[0], PEN_RADIUS, PEN_RADIUS)
                        self.setPixmapFit.emit(self.pattern.imageLabel, pixmap)
                        self.pattern.wait()
                    if DEBUG:
                        if len(self.rects) > 0 and len(self.circles) > 0:
                            break
                if DEBUG:
                    if len(self.rects) > 0 and len(self.circles) > 0:
                        break
                # if y & 0b1111==0b1111:
                #     self.setPixmapFit.emit(self.pattern.imageLabel, pixmap)
            painter.end()
            self.setPixmapFit.emit(self.pattern.imageLabel, pixmap)
            print('Done Detecting shape')

        def determineSizePos(self):
            self.sizePosPixmap = self.pattern.detectedEdgesPixmap.copy()
            # self.sizePosPixmap = self.pattern.convdImg.pixmap.copy()
            # self.sizePosPixmap = self.pattern.originImg.pixmap.copy()
            pixmap = self.sizePosPixmap

            PEN_WIDTH = 6
            painter = QPainter(pixmap)
            p = painter.pen()
            p.setWidth(PEN_WIDTH)
            p.setColor(QColorConstants.Cyan)
            painter.setPen(p)
            for rect in self.rects:
                center: QPointF = QPointF(rect.points[0])
                for p in rect.points[1:]:
                    p: QPointF
                    center += p
                center /= len(rect.points)

                closest = QPointF(rect.points[0])
                minDist = dist(closest, center)
                for p in rect.points[1:]:
                    p: QPointF
                    newDist = dist(p, center)
                    if newDist < minDist:
                        minDist = newDist
                        closest = p

                r = QPointF(minDist, minDist)

                painter.drawRect(QRectF(center-r, center+r))
                self.setPixmapFit.emit(self.pattern.imageLabel, pixmap)
                self.text.emit(
                    f'Rect: Center: ({center.x():.0f}, {center.y():.0f}),  Width: {minDist*2:.0f}')
                self.pattern.wait()

            p = painter.pen()
            p.setWidth(PEN_WIDTH)
            p.setColor(QColorConstants.Magenta)
            painter.setPen(p)
            for circle in self.circles:
                center = QPointF(circle.points[0])
                for p in circle.points[1:]:
                    p: QPointF
                    center += p
                center /= len(circle.points)

                closest = QPointF(circle.points[0])
                minDist = dist(closest, center)
                # for p in circle.points[1:]:
                #     p: QPointF
                #     newDist=dist(p,center)
                #     if newDist<minDist:
                #         minDist=newDist
                #         closest=p

                painter.drawEllipse(center, minDist, minDist)
                self.setPixmapFit.emit(self.pattern.imageLabel, pixmap)
                self.text.emit(
                    f'Circle: Center: ({center.x():.0f}, {center.y():.0f}),  Radius: {minDist:.0f}')
                self.pattern.wait()

    waitMode_WAIT = 0
    waitMode_Continue = 1

    def __init__(self, fname, mode=Pattern.Roberts):
        super().__init__(flags=Qt.Window)

        self.mutex = QMutex()
        self.waitCondition = QWaitCondition()
        self.waitMode = self.waitMode_Continue

        self.img = Image.open(fname)
        QApplication.processEvents()
        self.originImg = MImage()
        self.originImg.npimg = np.asarray(self.img)
        self.originImg.npimg = np.mean(self.originImg.npimg, axis=-1)
        self.originImg.npimg = self.originImg.npimg.astype(np.uint16)

        # self.originImg.pixmap =self.img.toqpixmap() # not working!
        # convert to np and back insread!
        def toqpixmap(img):
            obits = np.asarray(img)
            h, w, c = obits.shape
            print(obits.shape)
            pixmap = QPixmap(QImage(obits.tobytes(),
                                    w, h, w*c, QImage.Format.Format_BGR888))
            return pixmap
        self.originImg.pixmap = toqpixmap(self.img)

        self.showingImg = MImage()
        self.convdImg = MImage()

        self.mode = mode
        if mode == Pattern.Roberts:
            self.title = 'Roberts Operator'
            self.kernels = Pattern.Kernel.getRobertsOperators()
        elif mode == Pattern.Sobel:
            self.title = 'Sobel Operator'
            self.kernels = Pattern.Kernel.getSobelOperators()
        elif mode == Pattern.Prewitt:
            self.title = 'Prewitt Operator'
            self.kernels = Pattern.Kernel.getPrewittOperators()
        self.setWindowTitle(self.title)

        self.resize(WINDOW_SIZE)

        layout = QGridLayout()

        def addImageLabel(title, column):
            nonlocal layout
            imageLabel = ImageView()
            layout.addWidget(imageLabel, 0, column)

            textLabel = QLabel(title)
            textLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
            textLabel.setMaximumHeight(32)
            layout.addWidget(textLabel, 1, column)

            return imageLabel, textLabel

        self.imageLabel, self.textLabel = addImageLabel(
            'Detecting instances', 0)

        self.setLayout(layout)
        self.conv()
        self.show()

        self.mthread = self.WorkThread(self)
        self.mthread.setPixmapFit.connect(
            lambda imgView, pixmap: imgView.setPixmapFit(pixmap))
        self.mthread.plot.connect(self.plot)
        self.mthread.text.connect(self.text)
        self.mthread.start()

    def wait(self):
        if self.waitMode is self.waitMode_WAIT:
            self.mutex.lock()
            self.waitCondition.wait(self.mutex)
            self.mutex.unlock()

    def wake(self):
        self.waitCondition.wakeOne()

    def text(self, s):
        self.textLabel.setText(s)

    def keyPressEvent(self, event: QKeyEvent):
        # modifiers = QApplication.keyboardModifiers() # untrustable.  Use keyPressEvent instead
        if event.key() == Qt.Key.Key_Enter or event.key() == Qt.Key.Key_Return:
            self.waitMode = self.waitMode_WAIT
            self.wake()
        elif event.key() == Qt.Key.Key_M:
            self.waitMode = self.waitMode_Continue
            self.wake()

    def plot(self, data):
        plt.plot(data)
        plt.show()

    def conv(self):
        from scipy import signal
        h, w = self.originImg.npimg.shape

        bits = []
        for kernel in self.kernels:
            a = signal.convolve2d(
                self.originImg.npimg, kernel, boundary='symm', mode='same')
            # print(a.dtype) # int32
            # a = np.absolute(a)
            # a = np.clip(a, 0, 255)
            bits.append(a)

        bits = np.sqrt(bits[0]*bits[0]+bits[1]*bits[1])
        # print(np.min(bits), bits.max())
        bits = np.clip(bits, 0, 255)
        bits = bits.astype(np.uint8)
        self.convdImg.npimg = bits
        self.convdImg.pixmap = QPixmap(QImage(bits.tobytes(),
                                              w, h, w, QImage.Format.Format_Grayscale8))
        # self.imageLabel.setPixmapFit(self.originImg.pixmap)
        self.imageLabel.setPixmapFit(self.convdImg.pixmap)


if __name__ == '__main__':
    fname = None
    for argv in sys.argv[1:]:
        if argv == '-debug':
            DEBUG = True
        else:
            fname = argv

    main = Pattern(fname)
    splash.finish(main)
    app.exec_()
    sys.exit()
