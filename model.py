import torch
import torch.nn as nn
import numpy as np



class FeedForward(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0):
        super(FeedForward, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
        self.batch_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU()
        self.dropout = dropout
        self.dropout_layer = None
        if self.dropout != 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
    def forward(self, x):
        
        x = self.relu(self.batch_norm(self.linear_layer(x)))
        if self.dropout_layer != None:
            x = self.dropout_layer(x)
        return x 
        
        
class Attention(nn.Module):
    def __init__(self, feature_dim, bias):
        super(Attention, self).__init__()
        
        self.feature_dim = feature_dim
        self.model_dim = self.feature_dim // 4 
        self.W_Q = nn.Linear( self.feature_dim,  self.model_dim, bias=bias)
        self.W_K = nn.Linear( self.feature_dim,  self.model_dim, bias=bias)
        self.W_V = nn.Linear( self.feature_dim,  self.feature_dim, bias=bias)
        self.W_O = nn.Linear( self.feature_dim, self.feature_dim, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        allignment = self.softmax(torch.mm(Q, torch.t(K)) / (self.model_dim ** 0.5))
        atten = torch.mm( allignment, V)
        atten = self.W_O(atten)
        return atten


class PointEncoder(nn.Module):
    
    def __init__(self, in_features, feature_dim, out_featuers, dropout=0, bias=False):
        super(PointEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        self.in_features = in_features
        self.out_features = out_featuers
        
        self.embedding = nn.Sequential(FeedForward(self.in_features, self.feature_dim),
                                       FeedForward(self.feature_dim, self.feature_dim)
                                       )
        self.attention = Attention(self.feature_dim, bias=False)
        
    def forward(self, x):
        
        x = self.embedding(x)
        atten_1 = self.attention(x)
        atten_2 = self.attention(atten_1)
        atten_3 = self.attention(atten_2)
        atten_4 = self.attention(atten_3)
        out = torch.cat((atten_1, atten_2, atten_3, atten_4), dim=1)
        out = FeedForward(out.shape[1], self.out_features)(out)
        return out
        

class PointCloudClassifier(nn.Module):
     
    def __init__(self, in_features, feature_dim, out_featuers, k_size, num_classes):
        super(PointCloudClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.encoder = PointEncoder(in_features, feature_dim, out_featuers)
        
        self.max_pool = nn.MaxPool1d(k_size)
        self.avg_pool = nn.AvgPool1d(k_size)
        
        self.ff1 = FeedForward(16, 128, dropout=0.5)
        self.ff2 = FeedForward(128, 128, dropout=0.5)
        
        self.output_linear = nn.Linear(128, num_classes)
        self.linear = nn.Linear(128 * 128, self.num_classes)
        

    def forward(self, x):
        
        x = self.encoder(x)
        max_pool_out = self.max_pool(x)
        avg_pool_out = self.avg_pool(x)
        x = torch.cat((max_pool_out, avg_pool_out), axis=1)
        x = self.ff1(x)
        x = self.ff2(x)
        x = torch.flatten(x)
        x = self.linear(x)
        x = nn.Softmax(dim=-1)(x)
        return x
   
    
