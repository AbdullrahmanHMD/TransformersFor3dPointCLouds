import torch
import torch.nn as nn
import numpy as np


'''
    - This is an implementation of the original transformer
      that was represented in the paper "Attention is all you need".
    - The architecture will be used for 3D point cloud
'''

class LBRNN(nn.Module):
    
    def __init__(self, in_features, out_features, dropout=0, bias=False):
        super(LBRNN, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.bias = bias
        
        
        self.linear_layer = nn.Linear(self.in_features, self.out_features, bias=self.bias)
        self.batch_norm = nn.LayerNorm(self.out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = None
        
        nn.init.kaiming_normal_(self.linear_layer.weight.data)
        if bias is True:
            self.linear_layer.bias.data.fill_(0.1)
                    
      
        if self.dropout != 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        
    def forward(self, x):

        x = self.linear_layer(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        if self.dropout_layer != None:
            x = self.dropout_layer(x)
        return x 

# ############################################################### #

class Attention(nn.Module):
    def __init__(self, feature_dim, num_heads, eps=1e-6, bias=False):
        super(Attention, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = self.feature_dim // self.num_heads 
        self.eps = eps
        self.scale_factor = np.sqrt(1 / self.head_dim)

        assert (self.head_dim * self.num_heads == self.feature_dim)
        
        self.W_Q = nn.Linear( self.feature_dim,  self.feature_dim, bias=bias)
        self.W_K = nn.Linear( self.feature_dim,  self.feature_dim, bias=bias)
        self.W_V = nn.Linear( self.feature_dim,  self.feature_dim, bias=bias)
        self.W_O = nn.Linear( self.head_dim * self.num_heads, self.feature_dim, bias=bias)
        
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
                    


    def forward(self, x, q=None, k=None, v=None, mask=None):
        
        b, n, d = x.shape
        if q == None or k == None or v == None:    
            Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        else:
            Q, K, V = q, k ,V

        Q, K, V = map(lambda z: 
                    torch.reshape(z, (b,n,self.num_heads,self.head_dim)).
                    permute(0,2,1,3), [Q, K, V])

        out = torch.matmul(Q, torch.transpose(K, -1, -2)) / self.scale_factor 

        if mask is not None:
            out = out.masked_fill(mask == 0, float("-1e20"))
            
        
        out = self.softmax(out)
        out = out /  out.sum(axis=1, keepdim=True) + self.eps
        atten = torch.matmul(out, V)

        atten = atten.contiguous().view(b, n, -1)
        atten = self.W_O(atten)
        
        return atten
    


class TransformerBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads, out_dims, dropout=0):
        super(TransformerBlock, self).__init__()
        
        self.attention = Attention(embed_dim, num_heads)
        self.bn1 = nn.LayerNorm(embed_dim)
        self.bn2 = nn.LayerNorm(embed_dim)
        
        self.feedforward = nn.Sequential(LBRNN(embed_dim, embed_dim * out_dims, dropout=dropout),
                                         LBRNN(embed_dim * out_dims, embed_dim, dropout=dropout))
        
    
    def forward(self, x, mask=None):
        
        x = x + self.bn1(self.attention(x, mask=mask))
        x = x + self.bn2(self.feedforward(x))
        return x
        
        

class PctEncoder(nn.Module):
    
    def __init__(self, input_size, input_dims, embed_dim, out_dims, num_layers, num_heads, num_classes, dropout=0):
        super(PctEncoder, self).__init__()
        
        self.embedding = nn.Sequential(LBRNN(input_dims, embed_dim),
                                         LBRNN(embed_dim, embed_dim))
        self.trans_layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, 
                                 num_heads, 
                                 out_dims,
                                 dropout)
                for i in range(num_layers)
            ]
        )
        self.flatten_shape = input_size * embed_dim
        self.linear = nn.Linear(self.flatten_shape, num_classes)
        
        
    def forward(self, x, mask=None):
        
        x = self.embedding(x)
        for layer in self.trans_layers:
            x = layer(x, mask)
        
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x    
    



class PctDecoder(nn.Module):

    def __init__(self, input_size, input_dims, embed_dim, out_dim, num_layers, num_heads, dropout=0):
        super(PctDecoder, self).__init__()

        self.input_size = input_size
        self.input_dims = input_dims
        self.embed_dim = embed_dim
        self.out_dim = out_dim      
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        # ########################### #

        self.attention = Attention(self.embed_dim, self.num_heads)
        self.bn1 = nn.LayerNorm(self.embed_dim)
        self.transformer_block = TransformerBlock(embed_dim=self.embed_dim,
                                                  num_heads=self.num_heads,
                                                  out_dims=self.out_dim,
                                                  dropout=self.dropout)   
    
        self.linear = nn.Linear(40, 40)

    def forward(self, x):
        return x



















