import sys
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtUiTools import QUiLoader
from layout import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # self.ui_layout_file = QtCore.QFile("layout.ui")
        # self.ui_layout_file.open(QtCore.QFile.ReadOnly)
        # self.loader = QUiLoader()
        # # self.ui = self.loader.load()
        # self.loader.load(self.ui_layout_file, self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        ## Set Defaults
        self.gammaSliderChange(0)
        self.ui.gamma_slider.setValue(0)
        self.ccSliderChange(70)
        self.ui.cc_slider.setValue(70)
        
        ## set signals
        self.ui.gamma_slider.valueChanged.connect(self.gammaSliderChange)
        self.ui.cc_slider.valueChanged.connect(self.ccSliderChange)
        self.ui.source_directory_button.clicked.connect(self.sourceImageChange)
        self.ui.style_directory_button.clicked.connect(self.styleImageChange)
        self.ui.edit_directory_button.clicked.connect(self.outputDirectoryChange)
    
        # gamma_slider: QtWidgets.QSlider = window.gammaSlider
        # gamma_slider.valueChanged()
    
    def getGammaValue(self, raw, t: type = int):
        if t == str:
            return "0" if raw == 0 else "1e" + str(raw-1)
        
        return 0 if raw == 0 else 10**(raw-1)
    
    @QtCore.Slot(int)
    def gammaSliderChange(self, val):
        self.ui.gamma_label.setText(str(self.getGammaValue(val, str)))
        
    @QtCore.Slot(int)
    def ccSliderChange(self, val):
        self.ui.cc_label.setText(str(val/100))
        
    @QtCore.Slot()
    def sourceImageChange(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self)
        self.ui.source_directory_edit.setText(fileName[0])
        
    @QtCore.Slot()
    def styleImageChange(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self)
        self.ui.style_directory_edit.setText(fileName[0])
        
    @QtCore.Slot()
    def outputDirectoryChange(self):
        directoryName = QtWidgets.QFileDialog.getExistingDirectory(self)
        self.ui.output_directory_edit.setText(directoryName)
        
        

if __name__ == "__main__":
    print("Launch!")
    
    app = QtWidgets.QApplication([])
    window = MainWindow()
    
    # app = QtWidgets.QApplication([])
    # widget = MainWindow()
    # widget.resize(800, 600)
    # widget.show()
    window.show()
    sys.exit(app.exec())
    
# https://stackoverflow.com/questions/42046092/how-do-i-create-a-signal-to-open-a-qfiledialog-in-qt-designer