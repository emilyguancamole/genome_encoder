# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 23:14:34 2022

@author: Vermouth
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
class StructuredAutoencoderNet(nn.Module):
    ## Initialize the network
    def __init__(self, encoder_config, decoder_config, dropout_rate = 0):

        super().__init__()
        self.dropout_rate = dropout_rate
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        ## Save linear layer weights in a list
        self.weights_layer = []

        ## Generate encoder layer 
        index = 0
        self.encoder_layer = []
        # Loop over encoder layers 
        for i in range(len(self.encoder_config['dimension']) - 1): 
            self.encoder_layer.append(("linear" + str(index), nn.Linear(int(self.encoder_config['dimension'][i]), int(self.encoder_config['dimension'][i + 1]))))
            if i != len(self.encoder_config['dimension']) - 2: # if not the last layer
                self.encoder_layer.append(("ReLU" + str(index), nn.ReLU()))
                self.encoder_layer.append(("dropout" + str(index), nn.Dropout(p = dropout_rate)))
            index += 1
        
        for index, layer in enumerate(self.encoder_layer):
            if layer[0] == "linear":
                self.weights_layer.append(torch.nn.Parameter(layer[1].weight))
                self.encoder_layer[index][1].weight = self.weights_layer[-1]

        ## Generate decoder layer
        index = 0
        self.decoder_layer = []

        for i in range(len(self.decoder_config['dimension']) - 1): # Number of decoder layers
            # decoder_layer.append(("relu" + str(index),nn.ReLU()))
            if i != 0:
                self.decoder_layer.append(("dropout" + str(index), nn.Dropout(p = dropout_rate)))

            self.decoder_layer.append(("linear" + str(index), nn.Linear(int(self.decoder_config['dimension'][i]), int(self.decoder_config['dimension'][i + 1]))))
            if i != len(self.decoder_config['dimension']) - 2:
                self.decoder_layer.append(("ReLU" + str(index), nn.ReLU()))
            index += 1

        ## encoder_net and decoder_net
        self.encoder_net = nn.Sequential(OrderedDict(
          self.encoder_layer
        ))

        self.decoder_net = nn.Sequential(OrderedDict(
          self.decoder_layer
        ))
    
    # encode and decode function
    def encode(self, X):
        index = 0
        for layer in self.encoder_layer:
            if layer[0] == "linear":
                X = torch.nn.functional.linear(X, self.weights_layer[index])
                index += 1
            else:
                X = layer[1](X)
        # for layer in self.encoder_net:
        #     X = layer(X)
        return X 

    def decode(self, X):
        index = len(self.weights_layer) - 1
        for layer in self.decoder_layer:
            if layer[0] == "linear":
                X = torch.nn.functional.linear(X, self.weights_layer[index].t())
                index -= 1
            else:
                X = layer[1](X)
        # for layer in self.decoder_net:
        #     X = layer(X)
        return X

    # forward network
    def forward(self, X):
        X = self.encode(X)
        X = self.decode(X)
        
        return X 
def mse_loss(input_tensor, target_tensor):
    return ((input_tensor - target_tensor)**2).sum() / input_tensor.size()[0]

def StructuredMaskedAutoencoder(dataset, dropout_rate, N, encoder_dimension, decoder_dimension, train_epoch = 1000):
    ''' encoder_dimension: list of dimensions for encoder layers '''
    model = StructuredAutoencoderNet({'dimension' : encoder_dimension}, {'dimension' : decoder_dimension}, dropout_rate=dropout_rate)
    print(model)
    logging.basicConfig(filename="training_18241_512_40_2000_3e-05.log",level=logging.INFO)
    # Training configuration setting
    # criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-05)
    
    # Generate tensor list #* creates a list of tensors, where each tensor corresponds to gene expr data of a particular gene across all cells (cols)
    tensor_data_list = []

    for i in range(N): 
        tensor_data_list.append(torch.tensor(dataset.iloc[:,i].astype(np.float32).T)) 

    print("tensor_data_list len", len(tensor_data_list))
    print("tensor_data_list shape", tensor_data_list[0].shape)
    print("tensor_data_list shape", tensor_data_list[1].shape)

    
    # Training process
    train_losses = []
    validation_losses = []
    for epoch in range(int(train_epoch)):
        tensor_pred_list = []
        tensor_pred_list.append(model(tensor_data_list[0])) # first gene?
        tensor_pred_list.append(model(tensor_data_list[1])) # second gene? #? what is this model predicting? how is loss computed
        loss = 0       
        loss += mse_loss(tensor_data_list[0], tensor_pred_list[0])        
       #loss += criterion(tensor_data_list[0], tensor_pred_list[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}: Training loss: {loss.item()}")
        with torch.no_grad():
            val_loss = mse_loss(tensor_data_list[1], tensor_pred_list[1])
            if epoch % 100 == 0:
                print(f"Validation loss :{val_loss}")
        logging.info(f"Epoch {epoch+1}: Training Loss: {loss.item()}, Validation Loss :{val_loss}")

        # Early stopping
        if epoch > 100 and train_losses[-1] > train_losses[-2]:
            break

    # Embedding
    embedding_list = []
    recovered_list = []
    for i in range(N):
        embedding_list.append(model.encode(tensor_data_list[i]).detach().numpy())
        recovered_list.append(model(tensor_data_list[i]).detach().numpy())
    torch.save(model.state_dict(), "test-05.pt")
    # torch.save(model, "inVitro_sc_autoencoder_18241_512_40_2000_3e-05.pt")
    return embedding_list, recovered_list, train_losses

if __name__ == '__main__':
    # Load data
    dataset = pd.read_csv('Trevino_100.csv', index_col=0)
    print(dataset.head())
    N = len(dataset) # number of cells
    print("N",N)
    dropout_rate = 0
    encoder_dims = [N, 512, 40, 20] # maybe: 43340 cells = input. incorrect: 15469 genes = input
    decoder_dims = [20, N] # output of decoder is the same as original dim in order to calculate loss
    #* check if decoder needs to start with 40

    embedding_list, recovered_list, train_losses = StructuredMaskedAutoencoder(dataset, dropout_rate, N, encoder_dimension=encoder_dims, decoder_dimension=decoder_dims, train_epoch = 10000)
    # plot train loss and save it
    plt.plot(train_losses)
    plt.savefig("train_loss.png")
    


    # model = StructuredAutoencoderNet({'dimension' : encoder_dimension}, {'dimension' : decoder_dimension}, dropout_rate=dropout_rate)
    # model.encode()
    # model.decode

    # single encoding layer linear, single decode layer linear

    # one  input vector per cell, ie feed a single cell with all its 15000 genes


    # vanishing gradients! use relu instead of sigmoid

    # plot training error