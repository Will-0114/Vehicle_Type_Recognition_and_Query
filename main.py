import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from utils import image_processing, TDataSet, pytorch_utility
import os
import torchvision.models as models  ## use build-in model
import time
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import time
import cv2
import numpy as np
from View.ui import *
from data.Table import TableView

data = {'col1':['1','2','3','4'],
        'col2':['1','2','1','3'],
        'col3':['1','1','2','1']}
class AccuracyWindow(QWidget):
     def __init__(self):
         super(AccuracyWindow, self).__init__()
         self.resize(800, 600)
         # Label
         self.im = QPixmap("model/Accuracy_figure.png")
         self.label = QLabel(self)
         self.label.setGeometry(0, 0, 800, 600)
         self.label.setPixmap(self.im)
         self.label.setAlignment(QtCore.Qt.AlignTop) 
         self.label.setScaledContents(True)

class LossWindow(QWidget):
     def __init__(self):
         super(LossWindow, self).__init__()
         self.resize(800, 600)
         # Label
         self.im = QPixmap("model/Loss_figure.png")
         self.label = QLabel(self)
         self.label.setGeometry(0, 0, 800, 600)
         self.label.setPixmap(self.im)
         self.label.setAlignment(QtCore.Qt.AlignTop) 
         self.label.setScaledContents(True)
         
class PyQt_MVC_Main(QMainWindow):
    ## override the init function
    def __init__(self, parent = None):
        super(QMainWindow, self).__init__(parent)  ## inherit
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('PyQT 5 MVC example - by iVi Lab, Dept. IE&M, Taipei Tech')
        self.fn = ""
        self.html = ''
        self.A_window = AccuracyWindow()
        self.L_window = LossWindow()
        self.linkEvent()
        self.show()
        

    def linkEvent(self):
        ## add link between model and view
        self.ui.toolButton_Load_Image.clicked.connect(lambda: self.load_image())
##        self.ui.toolButton2_Load_Image.clicked.connect(lambda: self.load_image2())
        self.ui.toolButton_Delete_Image.clicked.connect(lambda: self.delete_image())
##        self.ui.toolButton2_Delete_Image.clicked.connect(lambda: self.delete_image2())
        self.ui.toolButton_Identify_Image.clicked.connect(lambda: self.identify_image())
        self.ui.toolButton_Catch.clicked.connect(lambda: self.catch())
        self.ui.actionAFigure.triggered.connect(self.A_window.show)
        self.ui.actionLFigure.triggered.connect(self.L_window.show)
        return
    
    def load_image(self):
        self.ui.Textlabel.setText("")
        filename = QFileDialog.getOpenFileName(self, 'Open Image File', './')
        if filename[0] =="":
            return None
        #print(filename)
        self.fn = filename[0]
        #print("Open image file: ", filename)
        #self.cvImage = cv2.imread(filename[0])
        self.cvImage=cv2.imdecode(np.fromfile(self.fn, dtype=np.uint8),-1)
        h,w,s = self.cvImage.shape
        #print(w,h,s)
        if (w > h):
            coe = 899/w
        elif (h >= w):
            coe = 599/h
        
        self.scale = coe #self.ui.spinBox_Scale.value() / 100
        self.cvImg = cv2.resize(self.cvImage, (0, 0), fx= self.scale, fy=self.scale)
        #print(type(self.cvImage))
        if self.cvImage is None:
            #print("No image is loaded")
            self.ui.label_Image.setText("No image is loaded")
            return
        self.display_img_on_label(self.cvImg)
        #self.isSegmented = False  ## 用來確認user 必須要做segment
        return self.cvImage ,self.fn
    
##    def load_image2(self):
##        self.ui.Textlabel.setText("")
##        filename = QFileDialog.getOpenFileName(self, 'Open Image File', './')
##        if filename[0] =="":
##            return None
##        #print(filename)
##        self.fn = filename[0]
##        #print("Open image file: ", filename)
##        #self.cvImage = cv2.imread(filename[0])
##        self.cvImage=cv2.imdecode(np.fromfile(self.fn, dtype=np.uint8),-1)
##        h,w,s = self.cvImage.shape
##        #print(w,h,s)
##        if (w > h):
##            coe = 899/w
##        elif (h >= w):
##            coe = 599/h
##        
##        self.scale = coe #self.ui.spinBox_Scale.value() / 100
##        self.cvImg = cv2.resize(self.cvImage, (0, 0), fx= self.scale, fy=self.scale)
##        #print(type(self.cvImage))
##        if self.cvImage is None:
##            #print("No image is loaded")
##            self.ui.label_Image2.setText("No image is loaded")
##            return
##        self.display_img_on_label2(self.cvImg)
##        #self.isSegmented = False  ## 用來確認user 必須要做segment
##        return self.cvImage ,self.fn
    
    def delete_image(self):
        if(self.fn == ""):
            self.ui.Textlabel.setText("You don't load any image")
            
        else:
            self.ui.Textlabel.setText("")
            self.fn = ""
            self.cvImage = ""
            self.ui.label_Image.setPixmap(QtGui.QPixmap("aoi.jpg"))
            self.ui.label_Image.setScaledContents(False)
            self.ui.label_Image.setFixedWidth(900)
            self.ui.label_Image.setFixedHeight(500)
            self.ui.label_Image.show()
            self.html = ''
        return self.fn
    
##    def delete_image2(self):
##        if(self.fn == ""):
##            self.ui.Textlabel3.setText("You don't Load any image")
##            
##        else:
##            self.ui.Textlabel3.setText("")
##            self.fn = ""
##            self.ui.label_Image2.setPixmap(QtGui.QPixmap("aoi.jpg"))
##            self.ui.label_Image2.setScaledContents(False)
##            self.ui.label_Image2.setFixedWidth(900)
##            self.ui.label_Image2.setFixedHeight(500)
##            self.ui.label_Image2.show()
##        return self.fn

    def display_img_on_label(self, cvImage):
        self.pixMap, w, h= self.convert_to_pixmap(cvImage)
        self.pixMap.scaled(w, h, QtCore.Qt.KeepAspectRatio)
        self.ui.label_Image.setPixmap(self.pixMap)
        self.ui.label_Image.setAlignment(QtCore.Qt.AlignTop) 
        self.ui.label_Image.setScaledContents(False)
        
        #self.ui.label_Image.setMinimumSize(1,1)
        self.ui.label_Image.show()
        return
    
##    def display_img_on_label2(self, cvImage):
##        self.pixMap, w, h= self.convert_to_pixmap(cvImage)
##        self.pixMap.scaled(w, h, QtCore.Qt.KeepAspectRatio)
##        self.ui.label_Image2.setPixmap(self.pixMap)
##        self.ui.label_Image2.setAlignment(QtCore.Qt.AlignTop) 
##        self.ui.label_Image2.setScaledContents(False)
##        #self.label_Image2.setMinimumSize(1,1)
##        
##        self.ui.label_Image2.show()
##        return
    

    def convert_to_pixmap(self, cvImg):
        img = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB) ## 改為 RGB: Pil_image
        height, width, byteValue = cvImg.shape
        byteValue = byteValue * width
        ## convert to pixmap
        mQImage = QtGui.QImage(img, width, height, byteValue, QtGui.QImage.Format_RGB888)
        pixMap = QtGui.QPixmap.fromImage(mQImage)
        return pixMap, width, height

    def identify_image(self):
        if(self.fn == ""):
            self.ui.Textlabel.setText("You don't load any image")
            
        else:
            class_list =["BMW_7_Series","BMW_X1", "BMW_X3","BMW_X4", "BMW_Z4",
                 "Ford_EcoSport", "Ford_Mondeo", "Ford_Mustang", "Ford_NEW FORD FOCUS", "Ford_Ranger",
                 "Honda_City", "Honda_CR-V","Honda_Fit", "Honda_HR-V","Honda_Odyssey",
                 "Nissan_ALTIMA", "Nissan_LIVINA", "Nissan_SENTRA", "Nissan_TIIDA", "Nissan_X-TRAIL",
                 "Toyota_ALTIS", "Toyota_COROLLA_CROSS", "Toyota_SIENTA", "Toyota_VIOS", "Toyota_YARIS",
                 "Volkswagen_The Golf R", "Volkswagen_The_Golf_280_TSI_R-Line", "Volkswagen_The_Golf_GTI_Performance", "Volkswagen_The_Golf_Variant_280_TSI_Highline", "Volkswagen_The_Tiguan_330_TSI_Comfortline"]
            t_start = time.perf_counter()
            model = pytorch_utility.load_full_model(filename="./model/Best_ResNet_Cars.pth")
            t_end = time.perf_counter()
            str1 = "Loading time: " + str(round((t_end- t_start), 3)) + " sec.\n"
            
            #print(str1)
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else: 
                device = torch.device("cpu")
            #filename = "./Truck/04617.jpeg"
            #filename = cv2.imread(self.fn)
            #filename = "./qiepspbz15.jpg"
            #print(filename)
            #cvImage = cv2.imread("./qiepspbz15.jpg")
            #label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = False)
            #print("Label/Probl: ", class_list[label], prob)
            #filename = "./Car/00002.jpeg"
            #cvImage = cv2.imread(filename, 1)
            t_start = time.perf_counter()
            label, prob = pytorch_utility.predict_cvImg(model, device,self.cvImage, isShow = False)
            t_end = time.perf_counter()
            #print("Label/Probl: ", class_list[label], prob)
            str2 = "Label/Probl: " + class_list[label] +" "+ str(prob)+"\n"
            
            #print(str2)
            #print("Predicting time: ", round((t_end- t_start), 3), " sec.")
            str3 = "Predicting time: " + str(round((t_end- t_start), 3)) + " sec."
            if prob > 0.8 :
                string = str1+str2+str3
                self.ui.Textlabel.setText(string)
                if class_list[label] =='BMW_7_Series':
                    self.html = 'bmw/7%20series/3889/overall'
                elif class_list[label] =='BMW_X1':
                    self.html = 'bmw/x1/3461/overall'
                elif class_list[label] =='BMW_X3':
                    self.html = 'bmw/x3/3829/overall'
                elif class_list[label] =='BMW_X4':
                    self.html = 'bmw/x4/3491/overall'
                elif class_list[label] =='BMW_Z4':
                    self.html = 'bmw/z4/3376/overall'
                elif class_list[label] =='Ford_EcoSport':
                    self.html = 'ford/ecosport/3642/overall'
                elif class_list[label] =='Ford_Mondeo':
                    self.html = 'ford/mondeo/3817/overall'
                elif class_list[label] =='Ford_Mustang':
                    self.html = 'ford/mustang/3634/overall'
                elif class_list[label] =='Ford_NEW FORD FOCUS':
                    self.html = 'ford/focus%205門/3840/overall'
                elif class_list[label] =='Ford_Ranger':
                    self.html = 'ford/ranger/3854/overall'
                elif class_list[label] =='Honda_City':
                    self.html = 'honda/city/3609/overall'
                elif class_list[label] =='Honda_CR-V':
                    self.html = 'honda/cr-v/3834/overall'
                elif class_list[label] =='Honda_Odyssey':
                    self.html = 'honda/odyssey/3518/overall'
                elif class_list[label] =='Honda_HR-V':
                    self.html = 'honda/hr-v/3816/overall'
                elif class_list[label] =='Honda_Fit':
                    self.html = 'honda/fit/3330/overall'
                elif class_list[label] =='Nissan_ALTIMA':
                    self.html = 'nissan/altima/3448/overall'
                elif class_list[label] =='Nissan_LIVINA':
                    self.html = 'nissan/livina/3668/overall'
                elif class_list[label] =='Nissan_SENTRA':
                    self.html = 'nissan/sentra/3861/overall'
                elif class_list[label] =='Nissan_TIIDA':
                    self.html = 'nissan/tiida/3693/overall';
                elif class_list[label] =='Nissan_X-TRAIL':
                    self.html = 'nissan/x-trail/3495/overall'
                elif class_list[label] =='Toyota_COROLLA_CROSS':
                    self.html = 'toyota/corolla%20cross/3830/overall'
                elif class_list[label] =='Toyota_SIENTA':
                    self.html = 'toyota/sienta/3480/overall'
                elif class_list[label] =='Toyota_VIOS':
                    self.html = 'toyota/vios/3630/overall'
                elif class_list[label] =='Toyota_YARIS':
                    self.html = 'toyota/yaris/3343/overall'
                elif class_list[label] =='Toyota_ALTIS':
                    self.html = 'toyota/corolla%20altis/3768/overall'
                elif class_list[label] =='Volkswagen_The Golf R':
                    self.html = 'volkswagen/golf%20r/3354/overall'
                elif class_list[label] =='Volkswagen_The_Golf_280_TSI_R-Line':
                    self.html = 'volkswagen/golf%20variant/3355/overall'
                elif class_list[label] =='Volkswagen_The_Golf_GTI_Performance':
                    self.html = 'volkswagen/golf/3493/overall'
                elif class_list[label] =='Volkswagen_The_Golf_Variant_280_TSI_Highline':
                    self.html = 'volkswagen/golf%20variant/3355/overall'
                elif class_list[label] =='Volkswagen_The_Tiguan_330_TSI_Comfortline':
                    self.html = 'volkswagen/tiguan/3694/overall'
                else:
                    self.html = ''
            else:
                self.ui.Textlabel.setText("Can not distinguish the car from the others")
                self.html = ''   
        return self.html

    def catch(self):
        # specify the url
        if (self.html == ''):
            self.ui.Textlabel.setText("You don't identify any image")
        else:
            urlpage = 'https://newcar.u-car.com.tw/'+self.html
            # query the website and return the html to the variable 'page'
            page = urllib.request.urlopen(urlpage)
            # parse the html using beautiful soup and store in variable 'soup'
            soup = BeautifulSoup(page, 'html.parser')
            # find results within table
            table = soup.find('table', attrs={'class': 'table_feature'})
            results = table.find_all('tr')
            Rows = []
            temp = []
            dataset = []
            Model = []
            Title = ['車型/規格', '售價(萬)', '排氣量(c.c)', '燃料', '引擎型式', '變速系統', '傳動方式', '最大馬力', '平均油耗 km/L(歐)', '相關稅費(元/年)']
            for i in range(1,10):
                Rows.append(format(Title[i],'10'))
            # loop over results
            for result in results:
                # find all columns per result
                data = result.find_all('td')
                # check that columns have data
                temp.clear()
                if len(data) == 0:
                    continue
                Model.append(data[0].getText().strip().ljust(15))
                for j in range(1,10):
                    text = data[j].getText().strip()
                    temp.append(text.ljust(15))
                dataset.append(temp[:])

            df = pd.DataFrame(dataset,index=Model,columns = Rows)
            df = df.to_string()

            self.ui.label2.setText(df)
        return 
    

    
        
def main():
    """
    主函数，用于运行程序
    :return: None
    """
    
    app = QtWidgets.QApplication(sys.argv)
    main = PyQt_MVC_Main()  # 注意修改为了自己重写的Dialog类
    #main.show()
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()

