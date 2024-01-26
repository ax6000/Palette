from abc import abstractmethod
import math
import glob
import os
from tqdm import tqdm,trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from .nn1D_v3 import (
    checkpoint,
    zero_module,
    normalization,
    count_flops_attn,
    gamma_embedding
)

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """
        
class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
class Multiresblock(EmbedBlock):
  def __init__(self,input_features : int, corresponding_unet_filters : int ,alpha : float =2.5)->None:
    """
        MultiResblock

        Arguments:
          x - input layer
          corresponding_unet_filters - Unet filters for the same stage
          alpha - 1.67 - factor used in the paper to dervie number of filters for multiresunet filters from Unet filters

        Returns - None

    """ 
    super().__init__()
    self.corresponding_unet_filters = corresponding_unet_filters
    self.alpha = alpha
    self.W = corresponding_unet_filters * alpha
    self.conv2d_bn_1x1 = Conv1d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5),
    kernel_size = 1,activation='None',padding = 0)

    self.conv2d_bn_3x3 = Conv1d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167),
    kernel_size = 3,activation='relu',padding = 1)
    self.conv2d_bn_5x5 = Conv1d_batchnorm(input_features=int(self.W*0.167),num_of_filters = int(self.W*0.333),
    kernel_size = 3,activation='relu',padding = 1)
    self.conv2d_bn_7x7 = Conv1d_batchnorm(input_features=int(self.W*0.333),num_of_filters = int(self.W*0.5),
    kernel_size = 3,activation='relu',padding = 1)
    self.batch_norm1 = nn.BatchNorm1d(int(self.W*0.5)+int(self.W*0.167)+int(self.W*0.333) ,affine=False)

  def forward(self,x: torch.Tensor)->torch.Tensor:

    temp = self.conv2d_bn_1x1(x)
    a = self.conv2d_bn_3x3(x)
    b = self.conv2d_bn_5x5(a)
    c = self.conv2d_bn_7x7(b)
    x = torch.cat([a,b,c],axis=1)
    x = self.batch_norm1(x)
    x = x +  temp
    x = self.batch_norm1(x)
    return x

class Conv1d_batchnorm(EmbedBlock):
  def __init__(self,input_features : int,num_of_filters : int ,kernel_size : int = 2,stride : int = 1, activation : str = 'relu',padding  : int= 0)->None:
    """
    Arguments:
      x - input layer
      num_of_filters - no. of filter outputs
      filters - shape of the filters to be used
      stride - stride dimension 
      activation -activation function to be used

    Returns - None
    """
    super().__init__()
    self.activation = activation
    self.conv1 = nn.Conv1d(in_channels=input_features,out_channels=num_of_filters,kernel_size=kernel_size,stride=stride,padding = padding)
    self.batchnorm = nn.BatchNorm1d(num_of_filters,affine=False)
    nn.init.kaiming_uniform_(self.conv1.weight.data)
    nn.init.zeros_(self.conv1.bias.data)
    print(self.conv1.weight.data)
  def forward(self,x : torch.Tensor,emb)->torch.Tensor:
    x = self.conv1(x)
    x = self.batchnorm(x)
    if self.activation == 'relu':
      return F.relu(x)
    else:
      return x


class Respath(EmbedBlock):
  def __init__(self,input_features : int,filters : int,respath_length : int)->None:
    """
    Arguments:
    input_features - input layer filters
    filters - output channels
    respath_length - length of the Respath
    
    Returns - None
    """
    super().__init__()
    self.filters = filters
    self.respath_length = respath_length
    self.conv2d_bn_1x1 = Conv1d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    kernel_size = 1,activation='None',padding = 0)
    self.conv2d_bn_3x3 = Conv1d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    kernel_size =3,activation='relu',padding = 1)
    self.conv2d_bn_1x1_common = Conv1d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    kernel_size =1,activation='None',padding = 0)
    self.conv2d_bn_3x3_common = Conv1d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    kernel_size =3,activation='relu',padding = 1)
    self.batch_norm1 = nn.BatchNorm1d(filters,affine=False)
    
  def forward(self,x : torch.Tensor,emb)->torch.Tensor:
    shortcut = self.conv2d_bn_1x1(x)
    x = self.conv2d_bn_3x3(x)
    x = x + shortcut    
    x = F.relu(x)
    x = self.batch_norm1(x)
    if self.respath_length>1:
      for i in range(self.respath_length):
        shortcut = self.conv2d_bn_1x1_common(x)
        x = self.conv2d_bn_3x3_common(x)
        x = x +  shortcut
        x = F.relu(x)
        x = self.batch_norm1(x)
      return x
    else:
      return x

class MultiResUnet(nn.Module):
#   def __init__(self,
#                channels : int,
#                filters : int =32,
#                nclasses : int =1)->None:
    def __init__(
        self,
        image_size,
        in_channel,
        inner_channel,
        out_channel,
        res_blocks,
        attn_res,
        dropout=0,
        channel_mults=(1, 2, 4, 8),
        conv_resample=True,
        use_checkpoint=False,
        use_fp16=False,
        
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,)->None:
        channels = in_channel
        filters = inner_channel
        nclasses=1

        """
        Arguments:
        channels - input image channels
        filters - filters to begin with (Unet)
        nclasses - number of classes

        Returns - None
        """
        super().__init__()
        # self.alpha = 1.67
        self.alpha = 2.5
        self.filters = filters
        self.nclasses = nclasses
        self.multiresblock1 = Multiresblock(input_features=channels,corresponding_unet_filters=self.filters)
        self.pool1 =  nn.MaxPool1d(2,stride= 2)
        self.in_filters1 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
        self.respath1 = Respath(input_features=self.in_filters1 ,filters=self.filters,respath_length=4)
        self.multiresblock2 = Multiresblock(input_features= self.in_filters1,corresponding_unet_filters=self.filters*2)
        self.pool2 =  nn.MaxPool1d(2, 2)
        self.in_filters2 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
        self.respath2 = Respath(input_features=self.in_filters2,filters=self.filters*2,respath_length=3)
        self.multiresblock3 = Multiresblock(input_features= self.in_filters2,corresponding_unet_filters=self.filters*4)
        self.pool3 =  nn.MaxPool1d(2, 2)
        self.in_filters3 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
        self.respath3 = Respath(input_features=self.in_filters3,filters=self.filters*4,respath_length=2)
        self.multiresblock4 = Multiresblock(input_features= self.in_filters3,corresponding_unet_filters=self.filters*8)
        self.pool4 =  nn.MaxPool1d(2, 2)
        self.in_filters4 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
        self.respath4 = Respath(input_features=self.in_filters4,filters=self.filters*8,respath_length=1)
        self.multiresblock5 = Multiresblock(input_features= self.in_filters4,corresponding_unet_filters=self.filters*16)
        self.in_filters5 = int(self.filters*16*self.alpha* 0.5)+int(self.filters*16*self.alpha*0.167)+int(self.filters*16*self.alpha*0.333)
        
        #Decoder path
        self.upsample6 = nn.ConvTranspose1d(in_channels=self.in_filters5,out_channels=self.filters*8,kernel_size=2,stride=2,padding = 0)  
        self.concat_filters1 = self.filters*8+self.filters*8
        self.multiresblock6 = Multiresblock(input_features=self.concat_filters1,corresponding_unet_filters=self.filters*8)
        self.in_filters6 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
        self.upsample7 = nn.ConvTranspose1d(in_channels=self.in_filters6,out_channels=self.filters*4,kernel_size=2,stride=2,padding = 0)  
        self.concat_filters2 = self.filters*4+self.filters*4
        self.multiresblock7 = Multiresblock(input_features=self.concat_filters2,corresponding_unet_filters=self.filters*4)
        self.in_filters7 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
        self.upsample8 = nn.ConvTranspose1d(in_channels=self.in_filters7,out_channels=self.filters*2,kernel_size=2,stride=2,padding = 0)  
        self.concat_filters3 = self.filters*2+self.filters*2
        self.multiresblock8 = Multiresblock(input_features=self.concat_filters3,corresponding_unet_filters=self.filters*2)
        self.in_filters8 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
        self.upsample9 = nn.ConvTranspose1d(in_channels=self.in_filters8,out_channels=self.filters,kernel_size=2,stride=2,padding = 0)  
        self.concat_filters4 = self.filters+self.filters
        self.multiresblock9 = Multiresblock(input_features=self.concat_filters4,corresponding_unet_filters=self.filters)
        self.in_filters9 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
        self.conv_final = Conv1d_batchnorm(input_features=self.in_filters9,num_of_filters = self.nclasses,
        kernel_size = 1,activation='None')
        
        

    def forward(self,x : torch.Tensor,gammas)->torch.Tensor:
        gammas = gammas.view(-1, )
        emb = self.cond_embed(gamma_embedding(gammas, self.inner_channel))
        x_multires1 = self.multiresblock1(x)
        x_pool1 = self.pool1(x_multires1)
        x_multires1 = self.respath1(x_multires1)
        x_multires2 = self.multiresblock2(x_pool1)
        x_pool2 = self.pool2(x_multires2)
        x_multires2 = self.respath2(x_multires2)
        x_multires3 = self.multiresblock3(x_pool2)
        x_pool3 = self.pool3(x_multires3)
        x_multires3 = self.respath3(x_multires3)
        x_multires4 = self.multiresblock4(x_pool3)
        x_pool4 = self.pool4(x_multires4)
        x_multires4 = self.respath4(x_multires4)
        x_multires5 = self.multiresblock5(x_pool4)
        up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
        x_multires6 = self.multiresblock6(up6)
        up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
        x_multires7 = self.multiresblock7(up7)
        up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
        x_multires8 = self.multiresblock8(up8)
        up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
        x_multires9 = self.multiresblock9(up9)
        conv_final_layer =  self.conv_final(x_multires9)
        # print(torch.mean(x_multires1).item(),torch.mean(x_multires2).item(),torch.mean(x_multires3).item(),torch.mean(x_multires4).item(),torch.mean(x_multires5).item(),torch.mean(x_multires6).item(),torch.mean(x_multires7).item(),torch.mean(x_multires8).item(),torch.mean(x_multires9).item(),)
        return conv_final_layer
