import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphConvolution



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.classes = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        #self.gc2 = GraphConvolution(nhid, nclass)
        self.gc2 = GraphConvolution(nhid, 2*nclass)
        self.dropout = dropout

    def forward(self, x, adj, model_index):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = x.view(x.shape[0], self.classes, 2) 
        x = x[:, model_index, :].squeeze(1)
        return F.log_softmax(x, dim=1)
