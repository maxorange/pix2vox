import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from gui_viewer import GUIViewer
from gui_draw import GUIDraw

class MainWindow(QMainWindow):

    def __init__(self, opt_engine, parent=None, win_size=450):
        QMainWindow.__init__(self, parent)
        self.widget = QWidget()
        self.opt_engine = opt_engine

        # voxel viewer widget
        self.frame = QFrame()
        self.viewerWidget = GUIViewer(self.frame, opt_engine)
        self.viewerWidget.setFixedSize(win_size, win_size)
        viewerBox = QVBoxLayout()
        viewerBox.addWidget(self.viewerWidget)
        self.frame.setLayout(viewerBox)

        # drawing widget
        self.drawWidget = GUIDraw(opt_engine, win_size)
        self.drawWidget.setFixedSize(win_size, win_size)

        # hbox2 widgets
        self.btnSave = QPushButton("Save")
        self.btnSample = QPushButton("Sample")
        self.btnDilation = QPushButton("Dilation")
        self.btnErosion = QPushButton("Erosion")
        self.btnColor = QRadioButton("Coloring")
        self.btnSketch = QRadioButton("Sketching")
        self.btnEraser = QRadioButton("Eraser")
        # group
        self.btnSketch.setChecked(True)
        btnGroup1 = QButtonGroup()
        btnGroup1.addButton(self.btnColor)
        btnGroup1.addButton(self.btnSketch)
        btnGroup1.addButton(self.btnEraser)
        # layouts
        btnLayout1 = QHBoxLayout()
        btnLayout1.addWidget(self.btnSave)
        btnLayout1.addWidget(self.btnSample)
        btnLayout1.addWidget(self.btnDilation)
        btnLayout1.addWidget(self.btnErosion)
        btnLayout1.setSpacing(30)
        btnWidget1 = QWidget()
        btnWidget1.setLayout(btnLayout1)
        btnWidget1.setFixedWidth(win_size)
        btnLayout2 = QHBoxLayout()
        btnLayout2.addWidget(self.btnColor)
        btnLayout2.addWidget(self.btnSketch)
        btnLayout2.addWidget(self.btnEraser)
        btnWidget2 = QWidget()
        btnWidget2.setLayout(btnLayout2)
        btnWidget2.setFixedWidth(win_size)

        # hbox3 widgets
        categoryLayout = self.category_layout()
        categoryWidget = QWidget()
        categoryWidget.setLayout(categoryLayout)
        scroll = QScrollArea()
        scroll.setWidget(categoryWidget)
        scroll.setFixedHeight(200)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.frame)
        hbox1.addWidget(self.drawWidget)
        hbox1.addStretch(1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(btnWidget1)
        hbox2.addWidget(btnWidget2)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(scroll)

        vbox1 = QVBoxLayout()
        vbox1.addLayout(hbox1)
        vbox1.addLayout(hbox2)
        vbox1.addLayout(hbox3)

        self.widget.setLayout(vbox1)
        self.setCentralWidget(self.widget)

        mainWidth = self.viewerWidget.width() + self.drawWidget.width() + 70
        mainHeight = self.viewerWidget.height() + 320
        self.setGeometry(200, 200, mainWidth, mainHeight)
        self.setFixedSize(self.width(), self.height())
        self.btnSave.clicked.connect(self.save_data)
        self.btnSample.clicked.connect(self.opt_engine.sample_z)
        self.btnDilation.clicked.connect(self.opt_engine.dilation)
        self.btnErosion.clicked.connect(self.opt_engine.erosion)
        self.btnColor.toggled.connect(self.drawWidget.use_color)
        self.btnSketch.toggled.connect(self.drawWidget.use_edge)
        self.btnEraser.toggled.connect(self.drawWidget.use_eraser)
        self.connect(self.opt_engine, SIGNAL('update_voxels'), self.viewerWidget.update_actor)
        self.opt_engine.start()

    def closeEvent(self, event):
        self.opt_engine.quit()
        self.opt_engine.model.sess.close()

    def category_layout(self):
        gridLayout = QGridLayout()
        gridLayout.setHorizontalSpacing(20)
        gridLayout.setVerticalSpacing(10)
        categories = np.genfromtxt("category.csv", usecols=(0,1,2), dtype=np.str, delimiter=',')
        for i, c in enumerate(categories):
            index, _, name = c
            label = QLabel(name)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            slider.valueChanged.connect(lambda value, j=int(index): self.opt_engine.set_label(float(value)/100, j))
            gridLayout.addWidget(label, i, 0)
            gridLayout.addWidget(slider, i, 1)
        return gridLayout

    def save_data(self):
        try:
            self.number
        except:
            self.number = 1
        self.drawWidget.uiColor.save()
        self.drawWidget.uiSketch.save()
        np.save("out/model-{0}.npy".format(self.number), self.opt_engine.get_3d_model())
        self.number += 1
