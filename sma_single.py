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
import scanpy as sc
import h5py

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
    logging.basicConfig(filename="training_18241_512_40_2000_3e-05.log",level=logging.INFO)
    # Training configuration setting
    # criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-05)
    
    # Generate tensor list #* creates a list of tensors, where each tensor corresponds to gene expr data of a particular gene across all cells (cols)
    tensor_data_list = []
    for i in range(N): # Loop over cells (columns)
        tensor_data_list.append(torch.tensor(dataset[:, i].astype(np.float32).T)) 
    
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


def read_data():
    
    with h5py.File('Trevino.h5', 'r') as f:
        expr_data = f['human_data/block1_values'][:] # contains ['human_data']
        print("Expression data read:", expr_data.shape)

    anndata = sc.AnnData(X=expr_data) # Convert expr data to Anndata for scanpy
    print("AnnData shape:", anndata)
    
    with h5py.File('Trevino.h5', 'r') as f:
        cell_names = f['human_data/axis0'][1:].astype(str) # Add cell names
 
    anndata.var_names = cell_names # adding cell names along columns to Anndata

    return anndata.X # expression data as numpy array


if __name__ == '__main__':
    # Load data
    print("Info: Loading data")
    # dataset = pd.read_csv('Trevino.csv', index_col=0)
    expr_data = read_data()
    N = expr_data.shape[1] # number of cells, 43340
    print("N",N)
    input_dim = expr_data.shape[0] # number of genes, 15469
    print("Input dim", input_dim)

    dropout_rate = 0
    train_epochs = 20000
    encoder_dims = [input_dim, 512, 20] # 15469 genes = input
    decoder_dims = [20, 512, input_dim] # output of decoder is the same as original dim in order to calculate loss
    print("Info: Training model")
    embedding_list, recovered_list, train_losses = StructuredMaskedAutoencoder(expr_data, dropout_rate, N, encoder_dimension=encoder_dims, decoder_dimension=decoder_dims, train_epoch=train_epochs)
    np.save("embedding_list_linear.npy", embedding_list) 
    # plot train loss and save it
    plt.plot(train_losses)
    plt.savefig("train_loss_linear.png")
    