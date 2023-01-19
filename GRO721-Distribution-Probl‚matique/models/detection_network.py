import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .classification_network import ClassificationNetwork

class DetectionNetwork(nn.Module):
    def __init__(self, input_channels, n_params):
        super(DetectionNetwork, self).__init__()
        self.n_params= n_params



        self.backbone= ClassificationNetwork(input_channels,n_params)

        self.outLayer = nn.Sequential(
             nn.Linear(256, 300),
             nn.BatchNorm1d(300),
             nn.ReLU(),
             nn.Linear(300,100),
             nn.BatchNorm1d(100),
             nn.ReLU(),
             nn.Linear(100,50),
             nn.BatchNorm1d(50),
             nn.ReLU(),
             nn.Linear(50, n_params*7)
             #nn.Sigmoid()
         )
        self.backbone.outLayer=self.outLayer

        self.sig=nn.Sigmoid()


    def forward(self, x):

        x= self.backbone (x)
        x=torch.reshape(x, (x.shape[0],3, 7))

        x1,x2=torch.split(x, 4,dim=2)
        x1= self.sig(x1)
        x2 = self.sig(x2)
        x=torch.cat((x1,x2),dim=2)
        return x





def stock(stuff,fileName='stuff'):
    f = open(fileName, "a")
    f.write(str(stuff))
    f.close()

LBoundingBox = lambda x,x_,y,y_,w,w_,h,h_: (x-x_)**2 +(y-y_)**2 \
        + (math.sqrt(w)-math.sqrt(w_))**2 + (math.sqrt(h)-math.sqrt(h_))**2

CERCLE, TRIANGLE, CROIX = 4,5,6
E, X, Y, H = 0, 1, 2, 3
class DetectionNetworkLoss(nn.Module):

    def __init__(self):
        super(DetectionNetworkLoss, self).__init__()

        self.coef = 5

        self.A = 1
        self.B = self.A* self.coef


    def forward(self,output,target):
        self.b_size=output.shape[0]
        #self.output=torch.reshape(output, (self.b_size,3, 7))
        self.output=output
        self.target=torch.zeros_like(self.output)
        I,self.num_of_BB,K = self.target.shape
        for i in range (I):
            for j in range(self.num_of_BB): # 0 à 2
                classe=int( target[i,j,4])
                miniList = [1 if k == classe else 0 for k in range(self.num_of_BB)]

                self.target[i,classe,:]= torch . cat ( (target[i,j,0:4],torch.FloatTensor(miniList)))

        stock(torch.mean(self.A * self.Lxywh(),dtype=torch.float),"Lxy")
        stock(torch.mean(self.B *self.L_classes(),dtype=torch.float), "Lclass")
        return torch.sum(self.A * self.Lxywh()+ self.B *self.L_classes() )

    def Lxywh(self):
        Lxywh_=torch.zeros(self.b_size)
        MSE=nn.MSELoss()
        for i in range(self.b_size):
            sum=0
            for j in range(self.num_of_BB) :
                sum+=MSE(self.output[i, :, [X, Y, H ]],
                   self.target[i, :, [X, Y, H ]])
                """sum+= LBoundingBox(
                    x=self.target[i,j,X],
                    y=self.target[i,j,Y],
                    w=self.target[i,j,H],
                    h=self.target[i,j,H],
                    x_=self.output[i,j,X],
                    y_=self.output[i,j,Y],
                    w_=self.output[i,j,H],
                    h_=self.output[i,j,H]
                )"""
            Lxywh_[i]= sum
        return Lxywh_

    def L_classes(self):
        L_classes_ = torch.zeros(self.b_size)
        CE=torch.nn.CrossEntropyLoss()
        BCE =torch.nn.BCELoss()
        for i in range(self.b_size):
            L_classes_[i] = CE(self.output[i, :, [E,CERCLE, TRIANGLE, CROIX]],
                                self.target[i, :, [E,CERCLE, TRIANGLE, CROIX]])

        return L_classes_

class FixedPredictorDetectionNetworkLoss:
    pass