# -*-coding: utf-8 -*-
"""
    This program is modified based on  ref: https://blog.csdn.net/guyuealian/article/details/88343924
    @Objective: set up pytorch image data in mini_batch with data balancing
                add in create the image_label_list by given directory
    @File   : TDataset.py
    @Author : Tien
    @E-mail : fctien@ntut.edu.tw
    @Date   : 20200515
"""
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

import sys
import random
import math
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QPushButton,QFileDialog
from PyQt5.QtGui import QPixmap,QImage
## LeNet
class Net(nn.Module):
    ## Note: 當input image size 改變時 需要修改 nn.linear 內的參數
    ##       需要重新計算一下，計算可以參考 calc_conv2.py
    def __init__(self, no_class= 10):
        super(Net, self).__init__()
        self.kernel_size1 = 5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5) ##
        ## nn.linear(input, output): input must be calulated
        self.fc1 = nn.Linear(16 * 61 * 61, 120) ## 16x61x61 請參考 calc_conv2.py
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, no_class)  ## output is 6

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) ## conv(5) 32 x 32 ==> 28 x 28 ## max_pool: 28 ==> 14 x 14
        x = self.pool(F.relu(self.conv2(x))) ## conv(5x5): 14 x 14 ==> 10 x 10 ## max_pool: 10 ==> 5 x 5
        x = x.view(-1, 16 * 61 * 61)  ## 16 kerne * 5x5
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_ResNet(train_image_dir = "./train", val_image_dir = "./test", no_epachs = 100, lr = 0.01,
                    type = "resnet18", num_classes = 30, pretrained = True, batch_size = 100, 
                    isDraw = True, model_name = "Best_ResNet_Cars.pth", split_ratio = 0.8):
    ## check if gpu available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = Torch.device("cpu")
    print("Training ResNet using ", device)
    
    ## select model ## choose one
    if type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif type == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif type == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif type == "resnet101":
        model = models.resnet101(pretrained=pretrained)
    elif type == "resnet152":
        model = models.resnet152(pretrained=pretrained)
    else:
        model = models.resnet18(pretrained=pretrained)

    ## Change the number_class feature ##############
    fc_features = model.fc.in_features
    #修改類別爲 num_classes = 6
    model.fc = nn.Linear(fc_features, num_classes)
    ################################################
    net = model.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    ## criterion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr= lr) #,
                      #momentum=0.8, weight_decay=5e-4)
    epoch_num= no_epachs   #总样本循环次数
    ## reset default image size = 224
    train_data = TDataSet.TDataset( image_dir=train_image_dir, resize_height= 224, resize_width = 224, repeat=1)
    total = len(train_data)
    import math  ## split the data (Note: num_data must be large, even data balancing is done in TDataSet)
    train_data, test_data = random_split(train_data, [math.ceil(total*split_ratio), total -math.ceil(total*split_ratio)])
    print("Split train data into: ", len(train_data), ":",len(test_data))
    #val_data = TDataSet.TDataset( image_dir=val_image_dir, resize_height= 224, resize_width = 224,repeat=1)
    # for image, label in train_data:
    #      print(image)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size= batch_size, shuffle=False)
    #val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True) ## for validation
    loss_list = list()
    acc_list = list()
    print("Start training proces ...")
    best_acc = 0
    for epoch in range(epoch_num):
        print("Epoch: ", epoch +1)
        t_start = time.perf_counter()
        correct = 0
        total = 0
        train_loss = 0
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader): ## __getitem__
            #inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device, dtype = torch.long)
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs.shape)
            #print(targets.shape)
            loss = criterion(outputs, targets) ## outputs:[batch, 3, row, col], target: [batch]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print("Batch: ", batch_idx, "==>", end="") ## no effect
            print(".", end = "", flush= True)
        
        train_acc = correct/total*100
        #print(train_loss, test_loss)
        t_end = time.perf_counter()
        print("==> Time span: ", round((t_end - t_start), 2), " sec.")
        print("==> Train Loss: ", round(train_loss, 4), " Train Acc: ", round(train_acc, 4))
        ############# testing #############
        test_loss, test_acc = pytorch_utility.test(epoch, net = net, testloader = test_loader, device = device, criterion = criterion)
        acc_list.append((train_acc, test_acc))
        loss_list.append((train_loss, test_loss))
        ## Save model ### Save checkpoint.
        if test_acc > best_acc:
            print('Saving best model..')
            # state = {
            #     'net': net.state_dict(),
            #     'acc': acc,
            #     'epoch': epoch,
            # }
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            # torch.save(state, './model/' + str(epoch+1) + "_ckpt.pth')
            pytorch_utility.save_full_model(net, filename = "./model/" + model_name)
            best_acc = test_acc
        print("==> Test loss:  ", round(test_loss, 4), " Test Acc: ", round(test_acc, 4), "===> Best Acc: ", round(best_acc, 4))
    ## draw loss and acc ##
    if isDraw:
        pytorch_utility.plot_loss(loss_list)
        pytorch_utility.plot_acc(acc_list)
    pytorch_utility.write_training_process(loss_list)
    return

def predict_example():
    t_start = time.perf_counter()
    model = pytorch_utility.load_full_model(filename="./model/Best_ResNet.pth")
    t_end = time.perf_counter()
    print(type(model), "Loading time: ", round((t_end- t_start), 3), " sec.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    filename = "./train/02_Rust/021.bmp"
    #cvImage = cv2.imread(filename, 1)
    label, prob = pytorch_utility.predict(model, device, filename = filename, resize_height = 224, resize_width = 224, isShow = True)
    print("Label/Probl: ", label, prob)
    filename = "./train/03_Small_Dots/051.bmp"

    t_start = time.perf_counter()
    label, prob = pytorch_utility.predict(model, device, filename = filename, resize_height = 224, resize_width = 224, isShow = False)
    t_end = time.perf_counter()
    print("Label/Probl: ", label, prob)
    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
    
    filename = "./train/04_Big_Dots/051.bmp"
    t_start = time.perf_counter()
    label, prob = pytorch_utility.predict(model, device, filename = filename, resize_height = 224, resize_width = 224, isShow = False)
    t_end = time.perf_counter()
    print("Label/Probl: ", label, prob)
    print("Predicting time: ", round((t_end- t_start), 3), " sec.")

    filename = "./train/06_Forging_Hurt/051.bmp"
    t_start = time.perf_counter()
    label, prob = pytorch_utility.predict(model, device, filename = filename, resize_height = 224, resize_width = 224, isShow = False)
    t_end = time.perf_counter()
    print("Label/Probl: ", label, prob)
    print("Predicting time: ", round((t_end- t_start), 3), " sec.")

    filename = "./train/07_Others/051.bmp"
    t_start = time.perf_counter()
    label, prob = pytorch_utility.predict(model, device, filename = filename, resize_height = 224, resize_width = 224, isShow = False)
    t_end = time.perf_counter()
    print("Label/Probl: ", label, prob)
    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
    return

def predict_cv_example():
    import cv2
    class_list =["BMW_7_Series","BMW_X1", "BMW_X3","BMW_X4", "BMW_Z4",
                 "Ford_EcoSport", "Ford_Mondeo", "Ford_Mustang", "Ford_NEW FORD FOCUS", "Ford_Ranger",
                 "Honda_City", "Honda_CR-V","Honda_Fit", "Honda_HR-V","Honda_Odyssey",
                 "Nissan_ALTIMA", "Nissan_LIVINA", "Nissan_SENTRA", "Nissan_TIIDA", "Nissan_X-TRAIL",
                 "Toyota_ALTIS", "Toyota_COROLLA_CROSS", "Toyota_SIENTA", "Toyota_VIOS", "Toyota_YARIS",
                 "Volkswagen_The Golf R", "Volkswagen_The_Golf_280_TSI_R-Line", "Volkswagen_The_Golf_GTI_Performance", "Volkswagen_The_Golf_Variant_280_TSI_Highline", "Volkswagen_The_Tiguan_330_TSI_Comfortline"]
    t_start = time.perf_counter()
    model = pytorch_utility.load_full_model(filename="./model/Best_ResNet_catdog.pth")
    t_end = time.perf_counter()
    print(type(model), "Loading time: ", round((t_end- t_start), 3), " sec.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    filename = "./qiepspbz15.jpg"
    cvImage = cv2.imread(filename, 1)
    #label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = True)
    #print("Label/Probl: ", class_list[label], prob)
    #filename = "./Car/00002.jpeg"
    #cvImage = cv2.imread(filename, 1)
    #t_start = time.perf_counter()
    label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = False)
    t_end = time.perf_counter()
    print("Label/Probl: ", class_list[label], prob)
    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
##    filename = "./train/cat/00002.jpeg"
##    cvImage = cv2.imread(filename, 1)
##    t_start = time.perf_counter()
##    label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = False)
##    t_end = time.perf_counter()
##    print("Label/Probl: ", class_list[label], prob)
##    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
##    filename = "./train/cat/00006.jpeg"
##    cvImage = cv2.imread(filename, 1)
##    t_start = time.perf_counter()
##    label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = False)
##    t_end = time.perf_counter()
##    print("Label/Probl: ", class_list[label], prob)
##    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
##    filename = "./train/cat/00016.jpeg"
##    cvImage = cv2.imread(filename, 1)
##    t_start = time.perf_counter()
##    label, prob = pytorch_utility.predict_cvImg(model, device, cvImage, isShow = False)
##    t_end = time.perf_counter()
##    print("Label/Probl: ", class_list[label], prob)
##    print("Predicting time: ", round((t_end- t_start), 3), " sec.")
    return

def evaluate_all():
    t_start = time.perf_counter()
    
    model = pytorch_utility.load_full_model(filename="./model/Best_ResNet.pth")
    t_end = time.perf_counter()
    print(type(model), "Loading time: ", round((t_end- t_start), 3), " sec.")
    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #else: 
    #    device = torch.device("cpu")
    pytorch_utility.evaluate_score(image_dir="./train", model=model, save_path = "./Misclassified")
    return

if __name__=='__main__':
    evaluate_all()
    #train_ResNet(train_image_dir = "./train", val_image_dir = "./test", no_epachs = 20, lr = 0.01,
                    #type = "resnet18", num_classes = 30, pretrained = True, batch_size = 20, isDraw = True) 
    #predict_cv_example()
    #app = QApplication(sys.argv)
    #mc=MyClass()
    #app.exec_()
    
