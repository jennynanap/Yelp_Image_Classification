#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:50:34 2019

@author: JosephVele
"""

#Mount Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

#Unzip photos on local drive
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


# Construct Data Loader
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
        labels = self.label_frame.iloc[idx,2] #Extract labels in the format ['1 2 3 4']
        labels = labels.split() #Make into list ['1','2','3','4']
        
        labels = list(map(int, labels)) #Convert to int
        #random.shuffle(labels)
        labels = [9] + labels + [10] # Include start and end
        labels = [x+1 for x in labels] # Add 1 to all labels so 0 has no meaning
        length = len(labels) 
        for i in range(11 - length): #Pad the labels (There are 9 unique labels)
            labels = labels + [0]
        
         
        target = torch.Tensor(labels).long()
        
        sample = {'image': image, 'labels': target, 'lengths': length}
        
        return sample
 
# Create  Datasets to be leveraged 
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

#Define the CNN RNN architecture

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the Alexnet Architecture modifying where appropriate"""
        super(EncoderCNN, self).__init__()
        alex_net = models.alexnet(pretrained=True)
        alex_net.classifier._modules['6'] = nn.Linear(4096, embed_size)

        
        self.alex_net = alex_net
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        
        features = self.alex_net(images)
        features = features.reshape(features.size(0), -1)
        #print(np.shape(features))
        #print(features.size())
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=11):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def execute_lstm(self, features, states=None):
        """Generate labels for given image """

        inputs = features.unsqueeze(1)
        for i in range(1):#self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)

        return outputs, states
    
    def embedding(self, features, states=None):
        """Embedd predicted label"""
        for i in range(1):#self.max_seg_length):
            inputs = self.embed(features)                       # inputs: (batch_size, embed_size)
        return inputs
    
#Hyper Parameter Tuning
        
vocab = ['<padding>','good_for_lunch','good_for_dinner','takes_reservations','outdoor_seating','restaurant_is_expensive',
         'has_alcohol','has_table_service','ambience_is_classy', 'good_for_kids','<start>', '<end>']

num_epochs = 64
total_step = len(yelpTrainLoader)
log_step = 10000
train_losses = []
validation_losses =[]
for epoch in range(num_epochs):
     
    embed_size = random.choice([512,1024,2048])
    hidden_size = random.choice([256,512,1024,2048])
    num_layers = random.choice([1,2])
    
    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

    learning_rate = round(np.exp(random.uniform(np.log(.0001), np.log(.01))),4) #pull geometrically
    w = round(np.exp(random.uniform(np.log(3.1e-7), np.log(3.1e-5))),10) #pull geometrically
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.alex_net.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = w)
    train_loss = 0
    validation_loss = 0
    
    encoder.train()
    decoder.train()
    for i, sample_batched in enumerate(yelpTrainLoader):
        
        # Set mini-batch dataset
        unsorted_images = sample_batched['image']
        unsorted_labels = sample_batched['labels']
        unsorted_lengths = sample_batched['lengths']
        sorted_length_index = sorted(range(len(unsorted_lengths)),key=unsorted_lengths.__getitem__,reverse=True)
        inputs =[]
        labels = []
        lengths = []
        for j in sorted_length_index:
            inputs.append(unsorted_images[j])
            labels.append(unsorted_labels[j])
            lengths.append(unsorted_lengths[j])
            
        inputs =torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        lengths = torch.stack(lengths)
        targets = pack_padded_sequence(labels , lengths, batch_first=True)[0].to(device)
        
        decoder.zero_grad()
        encoder.zero_grad()
        
        #print(labels)
        #print(targets)
        # Forward, backward and optimize
        features = encoder(inputs)
        outputs = decoder(features, labels , lengths)
        #print(outputs)
        #print(np.shape(outputs))
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # Print log info
        #if i % log_step == 0:
        #    print('Train Trial [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
        #          .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
        
    
    train_loss = train_loss/len(yelpTrainLoader.dataset)*1024 #1024 is the batch size
    train_losses.append([epoch,  embed_size, hidden_size, num_layers, learning_rate, w, train_loss ])
    
    print('Train Trial [{}/{}], Embedded Size: {}, Hidden Size: {}, Layers: {}, LR: {}, W: {}, Avg Loss: {}'
          .format(epoch, num_epochs, embed_size, hidden_size,
                  num_layers, learning_rate, w, train_loss ))

    #PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_encoder.pth'
    #torch.save(encoder.state_dict(), PATH)
    #PATH = '/content/gdrive/My Drive/Yelp_Project/alex_rnn_decoder.pth'
    #torch.save(decoder.state_dict(), PATH)
    train_losses_df = pd.DataFrame(train_losses)
    train_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/hypertrain_cnn_rnn_losses.csv')
    
    #Begin Validation
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
      for i, sample_batched in enumerate(yelpValidationLoader):
          # Set mini-batch dataset
          unsorted_images = sample_batched['image']
          unsorted_labels = sample_batched['labels']
          unsorted_lengths = sample_batched['lengths']
          sorted_length_index = sorted(range(len(unsorted_lengths)),key=unsorted_lengths.__getitem__,reverse=True)
          inputs =[]
          labels = []
          lengths = []
          for j in sorted_length_index:
              inputs.append(unsorted_images[j])
              labels.append(unsorted_labels[j])
              lengths.append(unsorted_lengths[j])

          inputs =torch.stack(inputs).to(device)
          labels = torch.stack(labels).to(device)
          lengths = torch.stack(lengths)
          targets = pack_padded_sequence(labels , lengths, batch_first=True)[0].to(device)
          #print(labels)
          #print(targets)
          # Forward, backward and optimize
          features = encoder(inputs)
          outputs = decoder(features, labels , lengths)
          #print(outputs)
          #print(np.shape(outputs))
          loss = criterion(outputs, targets)
          validation_loss += loss.item()
          
    validation_loss = validation_loss/len(yelpValidationLoader.dataset)*1024
    validation_losses.append([epoch,  embed_size, hidden_size, num_layers, learning_rate, w,  validation_loss ])
    
    print('Valid Trial [{}/{}], Embedded Size: {}, Hidden Size: {}, Layers: {}, LR: {}, W: {}, Avg Loss: {}'
          .format(epoch, num_epochs, embed_size, hidden_size,
                  num_layers, learning_rate, w, validation_loss ))
    validation_losses_df = pd.DataFrame(validation_losses)
    validation_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/hypervalid_cnn_rnn_losses.csv')
    
# Training the Model
    
 vocab = ['<padding>','good_for_lunch','good_for_dinner','takes_reservations','outdoor_seating','restaurant_is_expensive',
         'has_alcohol','has_table_service','ambience_is_classy', 'good_for_kids','<start>', '<end>']

embed_size = 2048
hidden_size = 2048
num_layers = 1
# Build the models
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)

learning_rate = 0.00025
w = 1.71e-06
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.alex_net.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay = w)


num_epochs = 40

train_losses = []
for epoch in range(num_epochs):
    train_loss = 0
    encoder.train()
    decoder.train()
    for i, sample_batched in enumerate(yelpTrainLoader):
        
        # Set mini-batch dataset
        unsorted_images = sample_batched['image']
        unsorted_labels = sample_batched['labels']
        unsorted_lengths = sample_batched['lengths']
        sorted_length_index = sorted(range(len(unsorted_lengths)),key=unsorted_lengths.__getitem__,reverse=True)
        inputs =[]
        labels = []
        lengths = []
        for j in sorted_length_index:
            inputs.append(unsorted_images[j])
            labels.append(unsorted_labels[j])
            lengths.append(unsorted_lengths[j])
            
        inputs =torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        lengths = torch.stack(lengths)
        targets = pack_padded_sequence(labels , lengths, batch_first=True)[0].to(device)
        
        decoder.zero_grad()
        encoder.zero_grad()
        
        #print(labels)
        #print(targets)
        # Forward, backward and optimize
        features = encoder(inputs)
        outputs = decoder(features, labels , lengths)
        #print(outputs)
        #print(np.shape(outputs))
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        
    train_loss = train_loss/len(yelpTrainLoader.dataset)*1024
    train_losses.append([epoch, train_loss ])
    
    print('Epoch [{}/{}], Embedded Size: {}, Hidden Size: {}, Layers: {}, LR: {}, W: {}, Avg Loss: {}'
          .format(epoch, num_epochs, embed_size, hidden_size,
                  num_layers, learning_rate, w, train_loss ))

    PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_encoder.pth'
    torch.save(encoder.state_dict(), PATH)
    PATH = '/content/gdrive/My Drive/Yelp_Project/alex_rnn_decoder.pth'
    torch.save(decoder.state_dict(), PATH)
    train_losses_df = pd.DataFrame(train_losses)
    train_losses_df.to_csv('/content/gdrive/My Drive/Yelp_Project/cnn_rnn_losses.csv')
    
    
# Define Beam Search
    
def beam_search(k, s, predicted, x, y, pred_sequence_list, prob_sequence_list):
  #Inputs Definitions:
  #k: Top labels to consider
  #s: current state
  #predicted: result of lstm (softmax)
  #x: current path of labels
  #y: current paht of probabilities 
  #prediction_paths: list of all label paths
  #probability_paths: list of all probability paths
    
    if predicted==11:
        #print(x)
        #print(y)
        pred_sequence_list.append(x)
        prob_sequence_list.append(y)
    else:
        inputs = decoder.embedding(predicted) 
        outputs, s = decoder.execute_lstm(inputs, s)
        scores = torch.softmax(outputs[0],dim=0)
        top_k_scores = scores.topk(k)[1].unsqueeze(0)
        top_k_probs = scores.topk(k)[0].unsqueeze(0)
        #print(top_k_scores)
        #print(top_k_probs)

        sequences = x.expand(k,len(x))
        prob_sequences = y.expand(k,len(x))
        #print(sequences)
        #print(top_k_scores[0][0].unsqueeze(0))
        #step =1

        for i in range(top_k_scores.size(1)):
            x = torch.cat((sequences[i], top_k_scores[0][i].unsqueeze(0) ))
            y = torch.cat((prob_sequences[i], top_k_probs[0][i].unsqueeze(0) ))
            
            predicted = x[len(x)-1].unsqueeze(0)

            if (x[len(x)-2]==10 and len(pred_sequence_list)<2) or predicted not in x[:-1]:
                #print('THis is predicted: ', x[:-1])
                #print('This is x: ',x)
                beam_search(k,s,predicted,x,y,pred_sequence_list,prob_sequence_list)

#Load Parameters
PATH = '/content/gdrive/My Drive/Yelp_Project/alex_cnn_encoder.pth'
encoder.load_state_dict(torch.load(PATH))
PATH = '/content/gdrive/My Drive/Yelp_Project/alex_rnn_decoder.pth'
decoder.load_state_dict(torch.load(PATH))

#Generate Sample Images with Captions

import sys 
import cv2 
sys.setrecursionlimit(3000)
photos = ['200115','200116','200117','200119','200121', '200122']
def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([227, 227], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
  
for i in photos:
    img_name = os.path.join('/content/train_photos', i  +'.jpg')
    

    image = load_image(img_name, transform).to(device)
    encoder.eval()
    decoder.eval()
    prediction_paths = []
    probability_paths = []
    # Encode - read the image features
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    output_sample, s = decoder.execute_lstm(encoder_out)
    _, predicted = output_sample.max(1)  

    sequences = predicted
    prob_sequences = torch.sigmoid(output_sample.max())
    prob_sequences

    beam_search(2, s, predicted, sequences, prob_sequences,prediction_paths, probability_paths)
    for i in range(len(prediction_paths)):
        if i ==0:
            best = np.prod(probability_paths[i].detach().cpu().numpy())
            index_val = i
        elif best < np.prod(probability_paths[i].detach().cpu().numpy()):
            best = np.prod(probability_paths[i].detach().cpu().numpy())
            index_val = i

    sampled_ids = prediction_paths[index_val].cpu().numpy()
    sampled_prob = best
    sampled_caption = []

    for word_id in sampled_ids:
        word = vocab[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption

    image = cv2.imread(img_name)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    plt.imshow(image)
    plt.show()
    print (sentence)
    print('Labels: {} \nPrediction_Path_Probability: {:.4f}'.format( sampled_ids,sampled_prob))


#Run Models on Test data using Beam Search

vocab = ['<padding>','good_for_lunch','good_for_dinner','takes_reservations',
         'outdoor_seating','restaurant_is_expensive',
         'has_alcohol','has_table_service','ambience_is_classy', 'good_for_kids','<start>', '<end>']


TP, TN, FP, FN = 0, 0, 0, 0
test_pred = []

photos = pd.read_csv('/content/gdrive/My Drive/test_photo_to_biz_ids.csv',
                     sep=",",dtype = 'str')

for i, j  in zip(photos['photo_id'], photos['label']):

    labels = j.split() #Make into list ['1','2','3','4']       
    labels = list(map(int, labels)) #Convert to int
    labels = [x+1 for x in labels] # Add 1 to all labels so 0 has no meaning
    
    #print(labels)
    
    img_name = os.path.join('/content/train_photos', i  +'.jpg')

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    encoder.eval()
    decoder.eval()
    
    # Prepare an image
    image = load_image(img_name, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    prediction_paths = []
    probability_paths = []
    # Encode - read the image features
    encoder_out = encoder(image_tensor)  # (1, enc_image_size, enc_image_size, encoder_dim)
    output_sample, s = decoder.execute_lstm(encoder_out)
    _, predicted = output_sample.max(1)  

    sequences = predicted
    prob_sequences = torch.sigmoid(output_sample.max())
    prob_sequences

    beam_search(2, s, predicted, sequences, prob_sequences,prediction_paths, probability_paths)
    for i in range(len(prediction_paths)):
        if i ==0:
            best = np.prod(probability_paths[i].detach().cpu().numpy())
            index_val = i
        elif best < np.prod(probability_paths[i].detach().cpu().numpy()):
            best = np.prod(probability_paths[i].detach().cpu().numpy())
            index_val = i

    sampled_ids = prediction_paths[index_val].cpu().numpy()
    
    pred = sampled_ids
    
    end = np.argwhere(pred==11)[0][0]
    start = np.argwhere(pred==10)[0][0]+1
    pred = pred[start:end]
    #print(pred)
    
    pred2 = np.zeros(9)
    labels2 = np.zeros(9)
    for i in pred:
        pred2[i-1]=1
    for i in labels:
        labels2[i-1]=1   
    test_pred.append([img_name,labels2,pred2])
    TP += ((pred2==1)&(labels2==1)).sum()
    TN += ((pred2==0)&(labels2==0)).sum()
    FP += ((pred2==1)&(labels2==0)).sum()
    FN += ((pred2==0)&(labels2==1)).sum()

accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
f1_score = 2 * (precision*recall)/(precision+recall)
print(TP,TN,FP,FN)
print('Test: Accuracy: {:.4f}\tPrecision: {:.4f}\t Recall: {:.4f}\t F1 Score: {:.4f}\n'.
          format(accuracy,  precision, recall, f1_score))
