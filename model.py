import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        
        features_resize = features.view(batch_size, -1, self.embed_size)
        captions_embeded = self.embed(captions)
        lstm_input = torch.cat((features_resize, captions_embeded),dim=1)
        lstm_input = lstm_input[:,0:-1,:]
        
        # hidden = self.init_hidden(batch_size, features.device)
        lstm_out, hidden = self.lstm(lstm_input)
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_size)
        out = self.fc(lstm_out)

        return out
            
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
       
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        
        return hidden

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        for i in range(max_len):
            out, states = self.lstm(inputs, states)
            out = out.contiguous().view(-1, self.hidden_size)
            out = self.fc(out)
            index = out.argmax(dim=1)
            caption.append(index.item())
            if index.item() == 1:
                break
            inputs = self.embed(index).view(1,1,self.embed_size)
            
        return caption