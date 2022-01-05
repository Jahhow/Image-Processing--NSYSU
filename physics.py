import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from numpy.random import *

# https://scipython.com/blog/two-dimensional-collisions/

class Ball:
    def __init__(self, x, y, r, vx, vy):
        self.pos=np.array([x,y], dtype=float)
        self.r = r
        self.m=r**2
        self.v=np.array([vx, vy], dtype=float)

    def isCollision(self, ball):
        newPos1=self.pos+self.v
        newPos2=ball.pos+ball.v
        newDistance=np.sum((newPos1-newPos2)**2)
        curDistance=np.sum((self.pos-ball.pos)**2)
        if newDistance>curDistance:
            return False
        return curDistance**.5 < self.r+ball.r

    def collide(self, ball):
        if self.isCollision(ball):
            # print('.',end='',flush=True)
            m1, m2 = self.m, ball.m
            M = m1 + m2
            xy1, xy2 = self.pos, ball.pos
            d = np.sum((self.pos-ball.pos)**2)
            v1, v2 = self.v, ball.v

            # u1 = v1 - 2*m2 * (v1-v2)*(xy1-xy2)**2 / M / d
            # u2 = v2 - 2*m1 * (v2-v1)*(xy2-xy1)**2 / M / d
            a=(v1-v2)
            b=(xy1-xy2)**2
            c=a*b / M / d
            u1 = v1 - 2*m2 * c
            u2 = v2 - 2*m1 * -c

            self.v = u1
            ball.v = u2

    def update(self, wallx, wally):
        x,y=self.pos
        vx,vy=self.v

        if x < self.r:
            x=self.r
            vx = abs(vx)
        elif x > wallx-self.r:
            x=wallx-self.r
            vx = -abs(vx)

        if y < self.r:
            y=self.r
            vy = abs(vy)
        elif y > wally-self.r:
            y=wally-self.r
            vy = -abs(vy)
        
        self.v=np.array([vx,vy],dtype=float)
        self.pos=np.array([x,y],dtype=float)
        self.pos+=self.v
        

    def draw(self,painter):
        painter.drawEllipse(QPointF(*self.pos), self.r, self.r)


class Canvas(QLabel):

    def __init__(self):
        super().__init__()
        self.pen_color = QColor('#aaa')
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAlignment(Qt.AlignTop)

        # Allow resize smaller than content pixmap
        sizePolicy = self.sizePolicy()
        sizePolicy.setVerticalPolicy(QSizePolicy.Ignored)
        sizePolicy.setHorizontalPolicy(QSizePolicy.Ignored)
        self.setSizePolicy(sizePolicy)

        # setFocus to receive keypress
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        self.balls = []
        for i in range(10):
            self.balls.append(self.randBall())

        qtimer = QTimer(self)
        qtimer.setInterval(20)
        qtimer.timeout.connect(self.mUpdate)
        qtimer.start()

    def randVelocity(self):
        return (random(2)-.5)*40

    def randBall(self):
        return Ball(randint(0, self.width()), randint(
                0, self.height()), randint(20, 80), *self.randVelocity())

    def mUpdate(self):
        # print('.',end='',flush=True)
        length=len(self.balls)
        for i in range(length-1):
            for j in range(i+1,length):
                self.balls[i].collide(self.balls[j])

        for ball in self.balls:
            ball.update(self.width(), self.height())

        pixmap = self.pixmap()
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        p = painter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        painter.setPen(p)

        for ball in self.balls:
            ball.draw(painter)

        painter.end()

        self.setPixmap(pixmap)

    # def mouseMoveEvent(self, e):
    #     self.curX=e.x()
    #     self.curY=e.y()
    #     self.mUpdate(e.x(),e.y())

    # def mouseReleaseEvent(self, e):
    #     self.last_x = None
    #     self.last_y = None

    def keyPressEvent(self, event):
        # modifiers = QApplication.keyboardModifiers() # untrustable.  Use keyPressEvent instead
        if event.key() == Qt.Key.Key_A:
            self.balls.append(self.randBall())
        elif event.key() == Qt.Key.Key_Z:
            if len(self.balls):
                self.balls.pop()
        else:
            for ball in self.balls:
                ball.v=self.randVelocity()

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
        self.setStyleSheet(
            "background-color: %s; border-style: solid;  border-width:1px;  border-radius:50px;" % color)
        self.setMask(QRegion(QRect(0, 0, 28, 28), QRegion.Ellipse))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Collisions")
        self.resize(600, 400)

        # UI --------------------------------------------------
        self.canvas = Canvas()

        # paletteLayout = QHBoxLayout()
        # paletteLayout.addWidget(QWidget(), 1)
        # self.add_palette_buttons(paletteLayout)
        # paletteLayout.addWidget(QWidget(), 1)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        # layout.addLayout(paletteLayout)

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
