from tkinter import X
from xml.etree.ElementPath import xpath_tokenizer
from xml.sax import xmlreader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
#-----------------------------
# encoder - decoder
# input 값 : feature map generated from a SfM 3D point cloud model

#-----------------------------
'''
axis 와 dim의 관계 파악
각 function 차이 파악
weight 옮겨보기
https://blog.pingpong.us/torch-to-tf-tf-to-torch/
'''
def weights_init(m):
    '''
    https://github.com/pytorch/examples/blob/main/dcgan/main.py#L95
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, mean=0, std=1)
        torch.nn.init.zeros_(m.bias)
class Batchnorm(nn.Module):
    '''
    input : NCHW
    bias : C
    '''
    def __init__(self, num_features, mean, var, bias, eps=1e-3, device=None, dtype=None) -> None:
        factory_kwargs = {'device':device, 'dtype':dtype}
        super(Batchnorm, self).__init__()
        self.num_features = num_features
        self.mean_ = mean
        self.var_ = var
        self.bias = bias
        self.eps = eps

        self.first_term = input/np.sqrt(var+eps)
        self.second_term = mean/np.sqrt(var+eps)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        first_term = input/np.sqrt(self.var_ + self.eps)
        second_term = self.mean_/np.sqrt(self.var_ + self.eps)
        return first_term - second_term + self.bias
# batch normalization and activation function after convolution layer in encoder
def ecBlock(input, running_mean, running_var, bias):
    output = F.batch_norm(input=input, running_mean=running_mean, running_var=running_var, weight=None, bias=bias, training=False, momentum=0., eps=1e-3)
    relu = nn.ReLU()
    output = relu(output)
    return output    

# batch normalization, dropout and activation function after convolution layer in decoder
def dcBlock(input, running_mean, running_var, bias, rate=0., act_func='relu'):
    # Bias term is added after batch normalization, not with the convolution layer
    # Therefore add bias term after batch normalization, except any weights
    output = F.batch_norm(input=input, running_mean=running_mean, running_var=running_var, weight=None, bias=bias, training=False, momentum=0., eps=1e-3)
    if act_func == 'relu':
        dropout = nn.Dropout(p=rate)
        act = nn.ReLU()
        output = dropout(output)
        output = act(output)
    elif act_func == 'sigmoid':
        act = nn.Sigmoid()
        output = act(output)
    return output

class VisibNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        if in_channels < 5:
        # SIFT descriptor X
            self.ech = [64,128,256,512,512,512]
        else:
        # SIFT descriptor O
            self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,1]

        self.rm_ec = [torch.zeros(ech) for ech in self.ech]
        self.rv_ec = [torch.ones(ech) for ech in self.ech]
        self.bias_ec = [torch.zeros(ech) for ech in self.ech]
        self.rm_dc = [torch.zeros(dch) for dch in self.dch]
        self.rv_dc = [torch.ones(dch) for dch in self.dch]
        self.bias_dc = [torch.zeros(dch) for dch in self.dch]

        # Encoder
        '''
        차후 nn.BatchNorm2D를 F.batch_norm 대신 사용할 것을 염두해두고 nn.Sequential로 작성
        '''
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
            # nn.BatchNorm2d(self.ech[0], eps=1e-3, momentum=0., affine=False, track_running_stats=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=False)
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[0], track_running_stats=False),
            # nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0]+self.ech[4], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[1], track_running_stats=False),
            # nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1]+self.ech[3], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[2], track_running_stats=False),
            # nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2]+self.ech[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[3], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3]+self.ech[1], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[4], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4]+self.ech[0], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[5], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5]+self.in_channels, self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[6], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[7], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(self.dch[8], track_running_stats=False),
            # nn.ReLU(inplace=False)
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Sigmoid()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, x):
        dx0 = x

        dx1 = self.down1(x)     # in_c  -> 256
        dx1 = ecBlock(dx1, self.rm_ec[0], self.rv_ec[0], self.bias_ec[0])

        dx2 = self.down2(dx1)   # 256   -> 256
        dx2 = ecBlock(dx2, self.rm_ec[1], self.rv_ec[1], self.bias_ec[1])

        dx3 = self.down3(dx2)   # 256   -> 256
        dx3 = ecBlock(dx3, self.rm_ec[2], self.rv_ec[2], self.bias_ec[2])

        dx4 = self.down4(dx3)   # 256   -> 512
        dx4 = ecBlock(dx4, self.rm_ec[3], self.rv_ec[3], self.bias_ec[3])

        dx5 = self.down5(dx4)   # 512   -> 512
        dx5 = ecBlock(dx5, self.rm_ec[4], self.rv_ec[4], self.bias_ec[4])

        dx6 = self.down6(dx5)   # 512   -> 512
        dx6 = ecBlock(dx6, self.rm_ec[5], self.rv_ec[5], self.bias_ec[5])
        x = dx6
        # # upsampling -> Conv -> BatchNorm -> ReLU -> concat
        # x = self.upsample(dx6)
        # x = self.up1(x)         # 512 -> 512
        # x = dcBlock(x, self.rm_dc[0], self.rv_dc[0], self.bias_dc[0])

        # x = torch.cat([dx5,x], dim=1)   # 512 -> 512+512
        # x = self.upsample(x)
        # x = self.up2(x)         # 1024 -> 512
        # x = dcBlock(x, self.rm_dc[1], self.rv_dc[1], self.bias_dc[1])

        # x = torch.cat([dx4,x], dim=1)   # 512 -> 512+512
        # x = self.upsample(x)
        # x = self.up3(x)         # 1024 -> 512
        # x = dcBlock(x, self.rm_dc[2], self.rv_dc[2], self.bias_dc[2])

        # x = torch.cat([dx3,x], dim=1)   # 512 -> 256+512
        # x = self.upsample(x)
        # x = self.up4(x)         # 768 -> 256
        # x = dcBlock(x, self.rm_dc[3], self.rv_dc[3], self.bias_dc[3])

        # x = torch.cat([dx2,x], dim=1)   # 256 -> 256+256
        # x = self.upsample(x)
        # x = self.up5(x)         # 512 -> 256
        # x = dcBlock(x, self.rm_dc[4], self.rv_dc[4], self.bias_dc[4])

        # x = torch.cat([dx1,x], dim=1)   # 256 -> 256+256
        # x = self.upsample(x)
        # x = self.up6(x)         # 512 -> 256
        # x = dcBlock(x, self.rm_dc[5], self.rv_dc[5], self.bias_dc[5])

        # x = torch.cat([dx0,x], dim=1)   # 256 -> 132+256
        # x = self.up7(x)         # 388 -> 128
        # x = dcBlock(x, self.rm_dc[6], self.rv_dc[6], self.bias_dc[6])

        # x = self.up8(x)         # 128 -> 64
        # x = dcBlock(x, self.rm_dc[7], self.rv_dc[7], self.bias_dc[7])

        # x = self.up9(x)         # 64 -> 32
        # x = dcBlock(x, self.rm_dc[8], self.rv_dc[8], self.bias_dc[8])

        # x = self.up10(x)        # 32 -> 1
        # x = dcBlock(x, self.rm_dc[9], self.rv_dc[9], self.bias_dc[9], act_func='sigmoid')
        return x  , dx1, dx2, dx3, dx4, dx5
         
class CoarseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,3]

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[0]),
            # nn.ReLU(inplace=False)
            #nn.Dropout()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[1]),
            # nn.ReLU(inplace=False)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[2]),
            # nn.ReLU(inplace=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[3]),
            # nn.ReLU(inplace=False)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[4]),
            # nn.ReLU(inplace=False)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[5]),
            # nn.ReLU(inplace=False)
        )
        
        # Decoder
        # tf.image.resize(images, size, method=ResizeMethod.BILINEAR)
        # Up(self, in_c, out_c, bilinear=True)
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[0]),
            # nn.ReLU(inplace=False)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[1]),
            # nn.ReLU(inplace=False)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[2]),
            # nn.ReLU(inplace=False)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[3]),
            # nn.ReLU(inplace=False)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[4]),
            # nn.ReLU(inplace=False)
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[5]),
            # nn.ReLU(inplace=False)
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5], self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[6]),
            # nn.ReLU(inplace=False)
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[7]),
            # nn.ReLU(inplace=False)
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[8]),
            # nn.ReLU(inplace=False)
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        dx1 = self.down1(x)     # in_c  -> 64
        dx2 = self.down2(dx1)   # 64    -> 128
        dx3 = self.down3(dx2)   # 128   -> 256
        dx4 = self.down4(dx3)   # 256   -> 512
        dx5 = self.down5(dx4)   # 512   -> 512
        dx6 = self.down6(dx5)   # 512   -> 512
        x = nn.Upsample(dx6, scale_factor=2, mode='nearest', align_corners=True)
        x = self.up1(x)
        x = torch.cat([dx6,x], dim=3)
        x = self.upsample(x)
        x = self.up2(x)
        x = torch.cat([dx5,x], dim=3)
        x = self.upsample(x)
        x = self.up3(x)
        x = torch.cat([dx4,x], dim=3)
        x = self.upsample(x)
        x = self.up4(x)
        x = torch.cat([dx3,x], dim=3)
        x = self.upsample(x)
        x = self.up5(x)
        x = torch.cat([dx2,x], dim=3)
        x = self.upsample(x)
        x = self.up6(x)
        x = torch.cat([dx1,x], dim=3)
        x = self.up7(x)
        x = self.up8(x)
        x = self.up9(x)
        x = self.up10(x)
        
class RefineNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ech = [256,256,256,512,512,512]
        self.dch = [512,512,512,256,256,256,128,64,32,3]

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, self.ech[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[0]),
            nn.LeakyReLU(inplace=False)
            #nn.Dropout()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(self.ech[0], self.ech[1], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[1]),
            nn.LeakyReLU(inplace=False)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(self.ech[1], self.ech[2], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[2]),
            nn.LeakyReLU(inplace=False)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(self.ech[2], self.ech[3], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[3]),
            nn.LeakyReLU(inplace=False)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(self.ech[3], self.ech[4], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[4]),
            nn.LeakyReLU(inplace=False)
        )
        self.down6 = nn.Sequential(
            nn.Conv2d(self.ech[4], self.ech[5], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.ech[5]),
            nn.LeakyReLU(inplace=False)
        )
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(self.ech[5], self.dch[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[0]),
            nn.LeakyReLU(inplace=False)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(self.dch[0], self.dch[1], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[1]),
            nn.LeakyReLU(inplace=False)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(self.dch[1], self.dch[2], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[2]),
            nn.LeakyReLU(inplace=False)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(self.dch[2], self.dch[3], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[3]),
            nn.LeakyReLU(inplace=False)
        )
        self.up5 = nn.Sequential(
            nn.Conv2d(self.dch[3], self.dch[4], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[4]),
            nn.LeakyReLU(inplace=False)
        )
        self.up6 = nn.Sequential(
            nn.Conv2d(self.dch[4], self.dch[5], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[5]),
            nn.LeakyReLU(inplace=False)
        )
        self.up7 = nn.Sequential(
            nn.Conv2d(self.dch[5], self.dch[6], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[6]),
            nn.LeakyReLU(inplace=False)
        )
        self.up8 = nn.Sequential(
            nn.Conv2d(self.dch[6], self.dch[7], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[7]),
            nn.LeakyReLU(inplace=False)
        )
        self.up9 = nn.Sequential(
            nn.Conv2d(self.dch[7], self.dch[8], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(self.dch[8]),
            nn.LeakyReLU(inplace=False)
        )
        self.up10 = nn.Sequential(
            nn.Conv2d(self.dch[8], self.dch[9], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        dx1 = self.down1(x)     # in_c  -> 64
        dx2 = self.down2(dx1)   # 64    -> 128
        dx3 = self.down3(dx2)   # 128   -> 256
        dx4 = self.down4(dx3)   # 256   -> 512
        dx5 = self.down5(dx4)   # 512   -> 512
        dx6 = self.down6(dx5)   # 512   -> 512
        x = self.upsample(dx6)
        x = self.up1(dx6)
        x = torch.cat([dx6,x], dim=3)
        x = self.upsample(x)
        x = self.up2(x)
        x = torch.cat([dx5,x], dim=3)
        x = self.upsample(x)
        x = self.up3(x)
        x = torch.cat([dx4,x], dim=3)
        x = self.upsample(x)
        x = self.up4(x)
        x = torch.cat([dx3,x], dim=3)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        x = self.up8(x)
        x = self.up9(x)
        x = self.up10(x)
    
# Convolutional layers of VGG16 for CoarseNet
# vgg16_model = models.vgg16(pretrained=True) 사용하면 됨. 개꿀
# relu1_1, relu2_2, relu3_3 pre-trained for image classification on the ImageNet 사용
class VGG16_loss(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vgg16 = models.vgg16(pretrained=True).features.to(device).eval()
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # relu1_1 : 1
        # relu2_2 : 8
        # relu3_3 : 15
        for x in range(2):
            self.slice1.add_module(str(x), vgg16[x])
        for x in range(2,9):
            self.slice2.add_module(str(x), vgg16[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg16[x])
        if not requires_grad:
            for param in self.parameters:
                param.requires_grad = False

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1, im2],0)
        im = self.normalize(im)

        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(im)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[0:3]:
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 + 1e-7) + (conf_sigma + 1e-7).log()
            if mask is not None:
                b, c, h,w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = F.avg_pool2d(mask, kernel_size=(sh, sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


# Discriminator network for training RefineNet
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def pred(self, inp):
        '''
        '''
        ncls = 2
        out = inp[0]
        
        # convolution layers
        cch = [256,256,256,512,512]
        for i, ch  in enumerate(cch):
            if i > 0 and i < len(inp):
                out = torch.cat([inp[i], out], dim=3)
            # Xavier Init


            out = nn.Sequential(
                nn.Conv2d(out, ch, kernel_size=3, stride=1, padding=1, padding_mode='same'),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU()
            )
        # return tf.reshape(out, [-1, ncls])
        return torch.reshape(out, [-1, ncls])

        

'''
tf.nn.conv2d(
    input, filters, strides, padding, data_format='NHWC', dilations=None,
    name=None
)
torch.nn.Conv2d(
    in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
    groups=1, bias=True, padding_mode='zeros', device=None, dtype=None
)

tf.nn.conv2d(
    out, filters=self.weights[], strides=[], padding='VALID'
)
torch.nn.Conv2d(
    out, 
)

tf.inp.get_shape().as_list()[-1]
-> list(inp.size())[-1]
'''