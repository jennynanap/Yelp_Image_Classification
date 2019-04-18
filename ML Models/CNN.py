#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:54:14 2019

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

#Alexnet requires 227 x 227 -Validation
yelpValidationSet = yelpDataset(text_file ='/content/gdrive/My Drive/validation_photo_to_biz_ids.csv',
                           root_dir = '/content/train_photos',
                          transform = transforms.Compose([transforms.Resize((227,227)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])]))

yelpValidationLoader = torch.utils.data.DataLoader(yelpValidationSet,batch_size=1024,shuffle=True, num_workers=0)

#Alexnet requires 227 x 227 - Test
yelpTestSet = yelpDataset(text_file ='/content/gdrive/My Drive/test_photo_to_biz_ids.csv',
                           root_dir = '/content/train_photos',
                          transform = transforms.Compose([transforms.Resize((227,227)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(
                                                              mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])]))

yelpTestLoader = torch.utils.data.DataLoader(yelpTestSet,batch_size=1024,shuffle=True, num_workers=0)


# Define the function for training, validation, and test
def alex_train(epoch, num_epochs,  train_losses, learning_rate, w):
    alex_net.train()
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
        outputs = alex_net(inputs)
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
    train_loss = train_loss/len(yelpTrainLoader.dataset)*1024 #1024 is the batch size
    train_losses.append([epoch, learning_rate, w, train_loss , TP, TN, FP, FN, accuracy, precision, recall, f1_score])
    # print statistics        
    print('Train Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'
          .format(epoch, num_epochs, learning_rate, w, train_loss, accuracy, f1_score))
    
def alex_valid(epoch, num_epochs, valid_losses, learning_rate, w):
    #Have our model in evaluation mode
    alex_net.eval()
    #Set losses and Correct labels to zero
    valid_loss = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i, sample_batched in enumerate(yelpValidationLoader,1):
            inputs = sample_batched['image'].to(device)
            labels = sample_batched['labels'].to(device)
            outputs = alex_net(inputs)
            loss = criterion(outputs.float(), labels.float())
            valid_loss += loss.item()
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
        valid_loss = valid_loss/len(yelpValidationLoader.dataset)*1024 #1024 is the batch size
        valid_losses.append([epoch, learning_rate, w, valid_loss , TP, TN, FP, FN, accuracy, precision, recall, f1_score])
        # print statistics        
        print('Valid Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'
          .format(epoch, num_epochs, learning_rate, w, valid_loss, accuracy, f1_score ))

def alex_test(epoch, num_epochs, pred_array,test_losses, learning_rate, w):
    #Have our model in evaluation mode
    alex_net.eval()
    #Set losses and Correct labels to zero
    test_loss = 0
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for i, sample_batched in enumerate(yelpTestLoader,1):
            inputs = sample_batched['image'].to(device)
            labels = sample_batched['labels'].to(device)
            paths = sample_batched['paths']
            outputs = alex_net(inputs)
            loss = criterion(outputs.float(), labels.float())
            test_loss += loss.item()
            pred = (torch.sigmoid(outputs).data > 0.5).int()
            #print(pred)
            labels = labels.int()
            #print(labels)
            pred_array.append([paths,test_loss,labels,pred])

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
        test_loss = test_loss/len(yelpTestLoader.dataset)*1024 #1024 is the batch size
        test_losses.append([epoch, learning_rate, w, test_loss , TP, TN, FP, FN, accuracy, precision, recall, f1_score])
        # print statistics        
        print('Valid Trial [{}/{}], LR: {}, W: {}, Avg Loss: {:.4f}, Accuracy: {:.4f}, F1 score: {:.4f}'
          .format(epoch, num_epochs, learning_rate, w, test_loss, accuracy, f1_score ))
        
#Hyper Parameter Tuning
alex_net = models.alexnet(pretrained=True)
for param in alex_net.parameters():
    param.requires_grad = False
alex_net.classifier._modules['6'] = nn.Linear(4096, 9)

train_losses = []
validation_losses = []
num_epochs = 8

for epoch in range(num_epochs):
  
  learning_rate = round(np.exp(random.uniform(np.log(.0001), np.log(.01))),4) #pull geometrically
  w = round(np.exp(random.uniform(np.log(3.1e-7), np.log(3.1e-5))),10) #pull geometrically
  
  #Reset Model per test
  alex_net = models.alexnet(pretrained=True)
  alex_net.classifier._modules['6'] = nn.Linear(4096, 9)
  alex_net.to(device)
  
  optimizer = torch.optim.Adam(alex_net.parameters(),lr=learning_rate, weight_decay = w)
  criterion = nn.BCEWithLogitsLoss()
  
  alex_train(epoch, num_epochs, train_losses, learning_rate, w)
  train_losses_df = pd.DataFrame(train_losses)
  train_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/hypertrain_cnn_losses.csv')
  
  alex_valid(epoch, num_epochs, validation_losses, learning_rate, w)
  validation_losses_df = pd.DataFrame(validation_losses)
  validation_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/hypervalid_cnn_losses.csv')
  
  
#Training
  
train_losses = []
num_epochs = 30
learning_rate = 0.0002
w = 1.73E-05
  
alex_net = models.alexnet(pretrained=True)
alex_net.classifier._modules['6'] = nn.Linear(4096, 9)
alex_net.to(device)
  
optimizer = torch.optim.Adam(alex_net.parameters(),lr=learning_rate, weight_decay = w)
criterion = nn.BCEWithLogitsLoss()
  
for epoch in range(num_epochs):
    
  alex_train(epoch, num_epochs, train_losses, learning_rate, w)
  train_losses_df = pd.DataFrame(train_losses)
  train_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/train_cnn_losses.csv')
  
  PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_baseline.pth'
  torch.save(alex_net.state_dict(), PATH)

#Testing
test_losses= []
test_pred = []
learning_rate = 0.0002
w = 1.73E-05
  
#Reset Model 
alex_net = models.alexnet(pretrained=True)
alex_net.classifier._modules['6'] = nn.Linear(4096, 9)
PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_baseline.pth'
alex_net.load_state_dict(torch.load(PATH))
alex_net.to(device)
  
optimizer = torch.optim.Adam(alex_net.parameters(),lr=learning_rate, weight_decay = w)
criterion = nn.BCEWithLogitsLoss()
alex_test(1, 1, test_pred, test_losses, learning_rate, w)
test_pred_df=pd.DataFrame(test_pred)
test_pred_df.to_csv('/content/gdrive/My Drive/Yelp_Project/cnn_predictions.csv')
print(test_losses)