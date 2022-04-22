import torch.nn as nn

from models.segnet.segnet import SegNet


class SegNetwork(SegNet):
    def __init__(self, input_nbr=3, label_nbr=22):
        super(SegNetwork, self).__init__(input_nbr, label_nbr)
        self.logsoftmax = nn.LogSoftmax(dim=2) 

    def forward(self, x):
        x11d = super().forward(x) 
        bs, nC, H, W = x11d.shape
        seg_pred = x11d.view(bs, nC//2, 2, H, W)
        seg_out = self.logsoftmax(seg_pred)
        return seg_out