import torch
import torch.nn as nn
import numpy as np




class FeedForward(nn.Module):
    
    """
        - FeedForward consists of a fully connected layer -> LayerNorm -> ReLU
        - if a dropout != 0, a dropout is added after ReLU 
    """
    
    def __init__(self, in_features, out_features, dropout=0, bias=False):
        super(FeedForward, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)
        nn.init.kaiming_normal_(self.linear_layer.weight.data)
        if bias is True:
            self.linear_layer.bias.data.fill_(0.1)
                    
        self.batch_norm = nn.LayerNorm(out_features)
        self.relu = nn.ReLU(inplace=True)
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
        
        """
        - A Self-Attentoin mechanism implemented from scratch
        """
        
        self.feature_dim = feature_dim
        self.model_dim = self.feature_dim // 4 
        self.W_Q = nn.Linear( self.feature_dim,  self.model_dim, bias=bias)
        self.W_K = nn.Linear( self.feature_dim,  self.model_dim, bias=bias)
        self.W_V = nn.Linear( self.feature_dim,  self.feature_dim, bias=bias)
        self.W_O = nn.Linear( self.feature_dim, self.feature_dim, bias=bias)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
                    
        
    def forward(self, x):
        
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        allignment = self.softmax(torch.bmm(Q, K.permute(0, -1, 1)) / (self.model_dim ** 0.5))
        atten = torch.bmm( allignment, V)
        atten = self.W_O(atten)
        return atten

   
class PointEncoder(nn.Module):
    
    
    
    def __init__(self, in_features, feature_dim, out_featuers, dropout=0, bias=False):
        super(PointEncoder, self).__init__()
        
        """
        - This is the encoder block. It consists of:
            - Embedding: Two cascaded FeedForward layers with dropout=0
            - Four self-attention layers concatenated together
            - FeedForward Layer 
        """
        
        self.feature_dim = feature_dim
        self.in_features = in_features
        self.out_features = out_featuers
        
        self.embedding = nn.Sequential(FeedForward(self.in_features, self.feature_dim),
                                       FeedForward(self.feature_dim, self.feature_dim)
                                       )
        self.attention = Attention(self.feature_dim, bias=bias)
        
    def forward(self, x):
        
        x = self.embedding(x)
        atten_1 = self.attention(x)
        atten_2 = self.attention(atten_1)
        atten_3 = self.attention(atten_2)
        atten_4 = self.attention(atten_3)
        out = torch.cat((atten_1, atten_2, atten_3, atten_4), dim=-1)
        out = FeedForward(out.shape[-1], self.out_features)(out)

        return out

    
    

class PointDecoder(nn.Module):
    
    def __init__(self, in_features, out_features, num_classes, dropout=0):
        
        """
        - The decoder block consists of:
            - Two Cascaded FeedForward layers
            - Linear layer with output dimenction of Nc.
        """
        super(PointDecoder, self).__init__()
        self._out_features = out_features
        self._in_features = in_features
        self.ff1 = FeedForward(self._in_features, self._out_features , dropout=dropout)
        self.ff2 = FeedForward(self._out_features, self._out_features, dropout=dropout)
        self.linear = nn.Linear(self._out_features, num_classes)
        
    
    def forward(self, x):
        x = self.ff2(self.ff1(x))
        x = self.linear(x)
        return x

    
class PointCloudClassifier(nn.Module):
     
    def __init__(self, num_encoders, model_dims, feature_dim, k_size, num_classes, dropout=0):
        super(PointCloudClassifier, self).__init__()
        
        #  Model Attributes
        self.num_classes = num_classes
        self.num_encoders = num_encoders
        self.feature_dim = feature_dim
        self.kernel_size = k_size
        self.encoders = []
        self.dims = model_dims
        for i in range(num_encoders):
            self.encoders.append(PointEncoder(self.dims[i], self.feature_dim, self.dims[i + 1]))
            self.feature_dim = 2 * self.feature_dim
            
        self.max_pool = nn.MaxPool1d(k_size)
        self.avg_pool = nn.AvgPool1d(k_size)
        
        self.decoder_out_features = 256
        self.dropout = dropout

        

    def forward(self, x):
        
        for encoder in self.encoders:
            x = encoder(x)
        x = torch.cat((self.max_pool(x), self.avg_pool(x)), axis=-1)
        x = torch.flatten(x, start_dim=1)
        x = PointDecoder(x.shape[1], self.decoder_out_features, self.num_classes, dropout=self.dropout)(x)
        return x
   
    
