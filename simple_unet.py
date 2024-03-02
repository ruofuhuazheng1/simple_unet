from torch import nn
import torch
from torch.nn import functional as F

class ConvolutionalBlock(nn.Module):
    """
    Convolutional Block that contains two convolutional layers, each followed by a batch normalization layer, 
    a dropout layer, and a LeakyReLU activation function.
    """
    def __init__(self, input_channels, output_channels):
        super(ConvolutionalBlock, self).__init__()
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, padding_mode='reflect', 
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, padding_mode='reflect', 
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
            )
    def forward(self, input_tensor):
        return self.convolutional_layers(input_tensor)
    
    
class DownSampler(nn.Module):
    """
    DownSampler block that reduces the spatial dimensions of the input tensor.
    """
    def __init__(self, input_channels):
        super(DownSampler, self).__init__()
        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(input_channels, input_channels,3,2,1,padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU()
            
            )
        
    def forward(self,input_tensor):
        return self.downsampling_layers(input_tensor)
    
    
class UpSampler(nn.Module):
    """
    UpSampler block that increases the spatial dimensions of the input tensor and concatenates it with a feature map.
    """
    def __init__(self, input_channels):
        super(UpSampler, self).__init__()
        self.upsampling_layer = nn.Conv2d(input_channels, input_channels//2,1,1)
        
    def forward(self,input_tensor, feature_map):
        upsampled_tensor = F.interpolate(input_tensor, scale_factor=2, mode='nearest')
        output_tensor = self.upsampling_layer(upsampled_tensor)
        return torch.cat((output_tensor,feature_map),dim=1)

    
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_block1=ConvolutionalBlock(1,64)
        self.downsampler1=DownSampler(64)
        self.conv_block2=ConvolutionalBlock(64, 128)
        self.downsampler2=DownSampler(128)
        self.conv_block3=ConvolutionalBlock(128,256)
        self.downsampler3=DownSampler(256)
        self.conv_block4=ConvolutionalBlock(256,512)
        self.downsampler4=DownSampler(512)
        self.conv_block5=ConvolutionalBlock(512,1024)
        self.upsampler1=UpSampler(1024)
        self.conv_block6=ConvolutionalBlock(1024,512)
        self.upsampler2=UpSampler(512)
        self.conv_block7=ConvolutionalBlock(512,256)
        self.upsampler3=UpSampler(256)
        self.conv_block8=ConvolutionalBlock(256,128)
        self.upsampler4=UpSampler(128)
        self.conv_block9=ConvolutionalBlock(128,64)
        
        self.output_layer = nn.Conv2d(64,1,3,1,1)
        self.threshold = nn.Sigmoid()

       
        
    def forward(self,input_tensor):
        conv1 = self.conv_block1(input_tensor)
        conv2 = self.conv_block2(self.downsampler1(conv1))
        conv3 = self.conv_block3(self.downsampler2(conv2))
        conv4 = self.conv_block4(self.downsampler3(conv3))
        conv5 = self.conv_block5(self.downsampler4(conv4))
      
        upsample1 = self.conv_block6(self.upsampler1(conv5,conv4))
        upsample2 = self.conv_block7(self.upsampler2(upsample1,conv3))
        upsample3 = self.conv_block8(self.upsampler3(upsample2,conv2))
        upsample4 = self.conv_block9(self.upsampler4(upsample3,conv1))
        
        return self.threshold(self.output_layer(upsample4))


         

