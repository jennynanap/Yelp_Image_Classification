#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:23:50 2019

@author: JosephVele
"""
from google.colab import drive
drive.mount('/content/gdrive')

!cp -r '/content/gdrive/My Drive/train_photos.zip' train_photos.zip
!unzip train_photos.zip

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import torchvision.models as models
import math 
from torch.nn.utils.rnn import pack_padded_sequence
%matplotlib inline
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class yelpDataset(torch.utils.data.Dataset):
    
    def __init__(self,text_file,root_dir, transform):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        
        self.name_frame = pd.read_csv(text_file,sep=",",usecols=range(1),dtype = 'str')
        self.label_frame = pd.read_csv(text_file,sep=",")
        self.root_dir = root_dir
        self.transform = transform
                                       
    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        #photoid = self.name_frame.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0]  +'.jpg')
        #print(img_name)
        image = Image.open(img_name)
        image = image.convert('RGB')
        image = self.transform(image) 
        labels = self.label_frame.iloc[idx,3:12].values
        labels = np.array(labels)
        labels= torch.from_numpy(labels.astype('int'))
        #print(labels)
        #labels = self.label_frame.iloc[idx,0]
        #labels = labels.reshape(-1, 2)
        sample = {'image': image, 'labels': labels, 'paths':img_name}
        
        return sample

#Alexnet requires 227 x 227 - Training
yelpTrainSet = yelpDataset(text_file ='/content/gdrive/My Drive/train_photo_to_biz_ids.csv',
                           root_dir = '/content/train_photos',
                          transform = transforms.Compose([transforms.Resize((227,227)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])]))

yelpTrainLoader = torch.utils.data.DataLoader(yelpTrainSet,batch_size=1024,shuffle=True, num_workers=0)


#Define SPP
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        #print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        
        #print(h_wid,w_wid,h_pad,w_pad)
        
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            #print("spp size:",spp.size())
        else:
            #print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

#Create the Model with SPP
class Alexnet_SPP(nn.Module):
    def __init__(self, labels):
        """Load the pretrained Alexnet and replace top fc layer."""
        super(Alexnet_SPP, self).__init__()
        alex_net = models.alexnet(pretrained=True)
        alex_net.classifier._modules['1'] = nn.Linear(5376,4096)
        alex_net.classifier._modules['6'] = nn.Linear(4096, 9)
        
        
        modules = list(alex_net.features.children())[:-1]    # delete the last maxpool layer.
        modules2 = list(alex_net.classifier.children())   
        self.conv = nn.Sequential(*modules)
        self.classifier = nn.Sequential(*modules2)
        self.output_num = [4,2,1]        
        
        
    def forward(self, images,batch_size):
        """Extract feature vectors from input images."""
        #with torch.no_grad():
        features = self.conv(images)
        self.batch=batch_size

        features = spatial_pyramid_pool(features,self.batch,
                                            [int(features.size(2)),
                                             int(features.size(3))],self.output_num)
        output = self.classifier(features)
        
        #print(np.shape(features))
        output = output.reshape(output.size(0), -1)

        return output

#Creat function for Training SPP   
def alex_train_spp(epoch, num_epochs,  train_losses, learning_rate, w):
    alex_net_spp.train()
    train_loss = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for i, sample_batched in enumerate(yelpTrainLoader,1):
        inputs = sample_batched['image'].to(device)
        labels = sample_batched['labels'].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = alex_net_spp(inputs,len(inputs))
        loss = criterion(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.item()
        pred = (torch.sigmoid(outputs).data > 0.5).int()
        #print(pred)
        labels = labels.int()
        #print(labels)
        
        TP += ((pred == 1) & (labels == 1) ).float().sum() #True Positive Count
        TN += ((pred == 0) & (labels == 0) ).float().sum() #True Negative Count
        FP += ((pred == 1) & (labels == 0) ).float().sum() #False Positive Count
        FN += ((pred == 0) & (labels == 1) ).float().sum() #False Negative Count
        #print('TP: {}\t TN: {}\t FP: {}\t FN: {}\n'.format(TP,TN,FP,FN) )  
    
    TP = TP.cpu().numpy()
    TN = TN.cpu().numpy()
    FP = FP.cpu().numpy()
    FN = FN.cpu().numpy()
    
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1_score = 2 * (precision*recall)/(precision+recall)
    train_loss = train_loss/len(yelpTrainLoader.dataset)*1024 # is the batch size
    train_losses.append([epoch, learning_rate, w, train_loss , TP, TN, FP, FN, accuracy, precision, recall, f1_score])
    # print statistics        
    print('Train Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'
          .format(epoch, num_epochs, learning_rate, w, train_loss, accuracy, f1_score))
    
 #Traing Model  
train_losses = []
num_epochs = 40
learning_rate = 0.0002
w = 1.73E-05
  

alex_net_spp = Alexnet_SPP(labels=9).to(device)
  
optimizer = torch.optim.Adam(alex_net_spp.parameters(),lr=learning_rate, weight_decay = w)
criterion = nn.BCEWithLogitsLoss()
 
  
for epoch in range(num_epochs):

  alex_train_spp(epoch, num_epochs, train_losses, learning_rate, w)
  train_losses_df = pd.DataFrame(train_losses)
  train_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/train_cnn_spp_losses_fixed.csv')
  
  PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_spp_baseline_fixed.pth'
  torch.save(alex_net_spp.state_dict(), PATH)

#Test Data
acc_data_test = []
test_losses = []
test_pred = []
num_epochs = 1
learning_rate = 0.0002
w = 1.73E-05

#
yelpTestSet = yelpDataset(text_file ='/content/gdrive/My Drive/test_photo_to_biz_ids.csv',
                           root_dir = '/content/train_photos',
                          transform = transforms.Compose([#transforms.Resize((227,227)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])]))

yelpTestLoader = torch.utils.data.DataLoader(yelpTestSet,batch_size=1,shuffle=True, num_workers=0)


alex_net_spp = Alexnet_SPP(labels=9).to(device)
optimizer = torch.optim.Adam(alex_net_spp.parameters(),lr=learning_rate, weight_decay = w)
criterion = nn.BCEWithLogitsLoss()
PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_spp_baseline_fixed.pth'
alex_net_spp.load_state_dict(torch.load(PATH))
alex_test(1,test_pred)
print(acc_data_test)
test_pred_df=pd.DataFrame(test_pred)
test_pred_df.to_csv('/content/gdrive/My Drive/Yelp_Project/cnn_spp_predictions.csv')


