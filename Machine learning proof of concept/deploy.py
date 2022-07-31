import os
import cv2
import sys
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch


IMG_SIZE = 100
data = []


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Convolution layers
        self.conv1 = nn.Conv2d(1, 32, 5)    #input 1 output 32 kernel 5
        self.conv2 = nn.Conv2d(32, 64, 5)   #input 32 output 64 kernel 5
        self.conv3 = nn.Conv2d(64, 128, 5)  #input 64 output 128 kernel 5


        x = torch.randn(100,100).view(-1,1,100,100) #shaped to fit input of conv layer 1 and size of image(100X100)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #using reshaped input to accept correct size.
        self.fc2 = nn.Linear(512, 2) # Output to two classes : Circle:0, Rectangle :1

    def convs(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            #reshape to size of input to the linear layer
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # reshape
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # activation layer
        return F.softmax(x, dim=1)



def make_test(pathstr):

    try:
        path = pathstr
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append([np.array(img)]) #append tensor data to list used for testing 
       
    except Exception as e:
        
        print(str(e)) # print error 
        pass



# print ('Argument List:', str(sys.argv)) ## test CLI arguments 

make_test(str(sys.argv[1])) # pass filename to fucntion to process as input 

net = Net()
net.load_state_dict(torch.load("model.pt")) #load trained model 
net.eval()

X = torch.Tensor([i[0] for i in data]).view(-1,100,100) #create tensor from data list in correct shape
X = X/255.0 # scale image valles from 0-255 to 0-1. only need black or white

with torch.no_grad():
    for i in tqdm(range(len(X))):
        net_out = net(X[i].view(-1, 1, 100, 100))[0]  #pass image into network
        predicted_class = torch.argmax(net_out)         # retrieve predicted class

if predicted_class.item() == 0:
    print("Prediction: Circle")
if predicted_class.item() == 1:
    print("Prediction: Rectangle")
