import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self,embed_size,train_CNN=False):
        super(CNN,self).__init__()
        self.train_CNN = train_CNN
        self.resnet = models.resnet152(pretrained=True, aux_logits=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,embed_size)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    
    def forw