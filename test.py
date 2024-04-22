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


def load_autoencoder_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def read_new_data():
    # Load your new dataset
    with h5py.File('Micali.h5', 'r') as f:
        new_expr_data = f['human_data/block1_values'][:]
    
    anndata = sc.AnnData(X=expr_data)

    with h5py.File('Micali.h5', 'r') as f:
        cell_names = f['human_data/axis0'][1:].astype(str) # Add cell names

    anndata.var_names = cell_names

    return anndata.X


def apply_autoencoder_on_new_data(model, new_data):
    # Convert new data to PyTorch tensor (if it's not already)
    new_data_tensor = torch.tensor(new_data.astype(np.float32))

    # Apply the autoencoder model on the new data
    with torch.no_grad():
        encoded_data = model.encode(new_data_tensor).detach().numpy()
        reconstructed_data = model(new_data_tensor).detach().numpy()

    return encoded_data, reconstructed_data


if __name__ == '__main__':
    expr_data = read_new_data()
    N = expr_data.shape[1] 
    input_dim = expr_data.shape[0]

    encoder_config = [input_dim, 512, 20] # 15469 genes = input
    decoder_config = [20, 512, input_dim]
    # Create an instance of your autoencoder model
    autoencoder_model = StructuredAutoencoderNet(encoder_config, decoder_config)

    # Load the saved weights into the model
    weights_path = "autoencoder_weights.pt"
    autoencoder_model = load_autoencoder_weights(autoencoder_model, weights_path)

    # Read your new dataset
    new_data = read_new_data()

    # Apply the autoencoder model on the new data
    encoded_data, reconstructed_data = apply_autoencoder_on_new_data(autoencoder_model, new_data)