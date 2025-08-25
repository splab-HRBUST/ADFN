import data_utils
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import librosa
import torch
from torch import nn
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from gru_am import VGGVox2
from scg_em_model import GatedRes2Net,SEGatedLinearConcatBottle2neck
from scipy.signal import medfilt
from scipy import signal
from dct_self import dct2,idct2
import data_utils
from torch.autograd import Variable
import torch.optim as optim
#from pick_data import get_data1,get_data2
from torch.nn.parameter import Parameter
import os
from dca2 import *
#from torchsummary import summary
#from sklearn.cross_decomposition import CCA
#from PCA import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




class cqt_mgd(nn.Module):
    def __init__(self, block, layers, num_classes, emb_dim1,emb_dim2,T,Q,Ax,Ay,
                 zero_init_residual=False):
        super(cqt_mgd, self).__init__()
        self.embedding_size1 = emb_dim1
        self.embedding_size2=emb_dim2
        self.num_classes = num_classes
        self.gru=VGGVox2(BasicBlock, [2, 2, 2, 2],emb_dim=self.embedding_size1)
        self.scg= GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, pretrained=False,loss='softmax')
        self.classifier_layer = nn.Linear(self.embedding_size2, self.num_classes)
        
        
        self.d_model = 512  
        self.wu = nn.Linear(self.d_model , self.d_model)
        self.wv = nn.Linear(self.d_model , self.d_model)
        self.ln1 = nn.Linear(self.d_model , self.d_model)
        
        
        self.Ax=torch.nn.Parameter(Ax)
        self.Ay=torch.nn.Parameter(Ay)
        self.A_x = nn.Linear(512, 512)
        #nn.init.xavier_normal_(self.A_x.weight, gain=1.0)
        nn.init.kaiming_normal_(
            self.A_x.weight, 
            mode='fan_in',      # 适应前向传播
            nonlinearity='relu' # 指定激活函数
            )
        self.A_y = nn.Linear(512, 512)
        #nn.init.xavier_normal_(self.A_y.weight, gain=1.0)
        nn.init.kaiming_normal_(
            self.A_y.weight, 
            mode='fan_in',      # 适应前向传播
            nonlinearity='relu' # 指定激活函数
            )
        '''
        self.T=torch.nn.Parameter(torch.transpose(T,dim0=0,dim1=1))
        self.Q=torch.nn.Parameter(torch.transpose(Q,dim0=0,dim1=1))
        self.fai_x=torch.nn.Parameter(torch.ones(512))
        self.fai_y=torch.nn.Parameter(torch.ones(512))
        self.fai1=torch.nn.Parameter(torch.ones(512))
        self.fai2=torch.nn.Parameter(torch.ones(512))

        self.E1=E1
        self.E11=E11
        self.E2=E2
        self.E22=E22
        self.E3=E3
        self.E33=E33
        '''

    def forward(self,x,y,device):
   
        
        y=self.gru(y) #(32,512)
        x=self.scg(x) #(32,512)
        x=torch.transpose(x,dim0=0,dim1=1) #(512,32)
        y=torch.transpose(y,dim0=0,dim1=1)#(512,32)
        
        
        x=torch.mm(self.Ax,x)
        y=torch.mm(self.Ay,y)
        #fai_x,fai_y,T,Q,m,mu_y,fai,E3=self.get_em_param(x,y,device)
        
        
        u = self.wu(torch.transpose(x,dim0=0,dim1=1))
        v = self.wv(torch.transpose(y,dim0=0,dim1=1))
            
        alpha = u+ v #wu*i + wv*j
        alpha = F.relu(alpha)

        out = F.relu(self.ln1(alpha))




        #out = torch.cat((x,y),dim=0)
        #print('out:{}'.format(out.shape))
        #emb=torch.transpose(out,dim0=0,dim1=1)#(32,256)
        out=self.classifier_layer(out)
        #print('out:{}'.format(out.shape))
        
        #return  x,y,fai_x,fai_y,T,Q,m,mu_y,fai,out,emb
        return  out


