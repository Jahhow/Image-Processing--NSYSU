import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from numpy.core.fromnumeric import size


class Canvas(QLabel):

    def __init__(self):
        super().__init__()
        self.curX=self.curY=None
        self.ctrl=False

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#aaa')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAlignment(Qt.AlignTop)
        self.setMode(self.Mode.FreeDraw)

        # Allow resize smaller than content pixmap
        sizePolicy = self.sizePolicy()
        sizePolicy.setVerticalPolicy(QSizePolicy.Ignored)
        sizePolicy.setHorizontalPolicy(QSizePolicy.Ignored)
        self.setSizePolicy(sizePolicy)

        # setFocus to receive keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    class Mode:
        FreeDraw=0
        InsertCircle=1
        InsertRectangle=2
        Crop=3

    def setMode(self, mode):
        self.mode=mode

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def mUpdate(self, x,y):
        if self.last_x is None:  # First event.
            if self.mode != self.Mode.FreeDraw:
                self.startPixmap=self.pixmap().copy()
            self.last_x = x
            self.last_y = y
            return  # Ignore the first time.

        # modifiers = QApplication.keyboardModifiers() # untrustable.  Use keyPressEvent instead
        if self.mode == self.Mode.FreeDraw:
            painter = QPainter(self.pixmap())
            p = painter.pen()
            p.setWidth(4)
            p.setColor(self.pen_color)
            painter.setPen(p)
            painter.drawLine(self.last_x, self.last_y, x, y)
            painter.end()
            self.update()

            # Update the origin for next time.
            self.last_x = x
            self.last_y = y
        elif self.mode == self.Mode.InsertCircle:
            pixmap=self.startPixmap.copy()
            painter = QPainter(pixmap)
            p = painter.pen()
            p.setWidth(4)
            p.setColor(self.pen_color)
            painter.setPen(p)
            
            if self.ctrl:
                painter.drawEllipse(QPoint(self.last_x,self.last_y), x-self.last_x, y-self.last_y)
            else:
                painter.drawEllipse(self.last_x, self.last_y, x-self.last_x, y-self.last_y)

            painter.end()
            self.setPixmap(pixmap)
        elif self.mode == self.Mode.InsertRectangle:
            pixmap=self.startPixmap.copy()
            painter = QPainter(pixmap)
            p = painter.pen()
            p.setWidth(4)
            p.setColor(self.pen_color)
            painter.setPen(p)

            if self.ctrl:
                painter.drawRect(2*self.last_x-x, 2*self.last_y-y, (x-self.last_x)*2, (y-self.last_y)*2)
            else:
                painter.drawRect(self.last_x, self.last_y, x-self.last_x, y-self.last_y)

            painter.end()
            self.setPixmap(pixmap)
        elif self.mode == self.Mode.Crop:
            pixmap=self.startPixmap.copy()
            painter = QPainter(pixmap)
            # p = painter.pen()
            # p.setWidth(4)
            # p.setColor(Qt.GlobalColor.transparent)
            # painter.setPen(p)
            painter.setCompositionMode (QPainter.CompositionMode_Source)
            painter.fillRect(self.last_x, self.last_y, x-self.last_x, y-self.last_y, QColor(0,0,0,0))
            # painter.setCompositionMode (QPainter::CompositionMode_SourceOver);

            painter.end()
            self.setPixmap(pixmap)

    def mouseMoveEvent(self, e):
        self.curX=e.x()
        self.curY=e.y()
        self.mUpdate(e.x(),e.y())

    def keyPressEvent(self, event):
        # modifiers = QApplication.keyboardModifiers() # untrustable.  Use keyPressEvent instead
        if event.key() == Qt.Key_Control:
            self.ctrl=True
        if self.curX is not None and self.last_x is not None:
            self.mUpdate(self.curX, self.curY)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrl=False
        if self.curX is not None and self.last_x is not None:
            self.mUpdate(self.curX, self.curY)

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        if self.pixmap() is None:
            pixmap = QPixmap(self.width(), self.height())
            pixmap.fill(Qt.GlobalColor.transparent)
            self.setPixmap(pixmap)
            return

        w, h = self.pixmap().width(), self.pixmap().height()
        nw, nh = self.width(), self.height()

        def qimageToNumpy(qimage, w, h):
            bits = qimage.bits().asarray(h*w*4)
            bits = np.array(bits, dtype=np.uint8)
            bits = bits.reshape([h, w, 4])
            return bits
        oldPic = qimageToNumpy(self.pixmap().toImage(), w, h)
        newPic = np.zeros([nh, nw, 4], dtype=np.uint8)
        newPic[:min(h, nh), :min(w, nw)] = oldPic[:min(h, nh), :min(w, nw)]
        pixmap = QPixmap(QImage(newPic.tobytes(), nw, nh, 4 *
                         nw, QImage.Format.Format_RGBA8888))
        self.setPixmap(pixmap)


COLORS = [
    # 17 undertones https://lospec.com/palette-list/17undertones
    '#000000', '#141923', '#414168', '#3a7fa7', '#35e3e3', '#8fd970', '#5ebb49',
    '#458352', '#dcd37b', '#fffee5', '#ffd035', '#cc9245', '#a15c3e', '#a42f3b',
    '#f45b7a', '#c24998', '#81588d', '#bcb0c2', '#ffffff',
]


class QPaletteButton(QPushButton):

    def __init__(self, color):
        super().__init__()
        self.color = color

        self.setFixedSize(QSize(32, 32))
        self.setStyleSheet("background-color: %s; border-style: solid;  border-width:1px;  border-radius:50px;" % color)
        self.setMask(QRegion(QRect(0,0,28,28), QRegion.Ellipse))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draw")
        self.resize(600, 400)
        # self.mode = self.Mode.FreeDraw

        # Menu ------------------------------------------------

        menuBar = self.menuBar()
        # Creating menus using a title
        # fileMenu = menuBar.addMenu("&File")
        insertMenu = menuBar.addMenu("&Edit")
        # helpMenu = menuBar.addMenu("&Help")

        insertFreeDrawAction = QAction("&Free Draw", self)
        insertFreeDrawAction.triggered.connect(lambda: self.canvas.setMode(Canvas.Mode.FreeDraw))
        insertMenu.addAction(insertFreeDrawAction)

        insertCircleAction = QAction("&Circle", self)
        insertCircleAction.triggered.connect(self.insertCircle)
        insertMenu.addAction(insertCircleAction)

        insertRectAction = QAction("&Rectangle", self)
        insertRectAction.triggered.connect(self.insertRectangle)
        insertMenu.addAction(insertRectAction)

        insertCropAction = QAction("&Crop", self)
        insertCropAction.triggered.connect(lambda: self.canvas.setMode(Canvas.Mode.Crop))
        insertMenu.addAction(insertCropAction)

        # UI --------------------------------------------------

        self.canvas = Canvas()

        paletteLayout = QHBoxLayout()
        paletteLayout.addWidget(QWidget(),1)
        self.add_palette_buttons(paletteLayout)
        paletteLayout.addWidget(QWidget(),1)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(paletteLayout)

        mainWidget = QWidget()
        mainWidget.setLayout(layout)
        self.setCentralWidget(mainWidget)

    def add_palette_buttons(self, layout):
        for c in COLORS:
            b = QPaletteButton(c)
            b.pressed.connect(lambda c=c: self.canvas.set_pen_color(c))
            layout.addWidget(b)

    def insertCircle(self, event):
        # print('insertCircle')
        self.canvas.setMode(Canvas.Mode.InsertCircle)
        # self.mode = self.Mode.InsertCircle

    def insertRectangle(self, event):
        # print('insertRectangle')
        self.canvas.setMode(Canvas.Mode.InsertRectangle)
        # self.mode = self.Mode.InsertRectangle


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
