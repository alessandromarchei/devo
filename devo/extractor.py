import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

DIM=32 # default 32

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, dim=DIM, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        self.dim = dim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.dim)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, self.dim, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = self.dim
        self.layer1 = self._make_layer(self.dim,  stride=1)
        self.layer2 = self._make_layer(2*self.dim, stride=2)
        self.layer3 = self._make_layer(4*self.dim, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*self.dim, output_dim, kernel_size=1)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class BasicEncoder4(nn.Module):
    def __init__(self, output_dim=128, dim=DIM, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder4, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        self.dim = dim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.dim)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, self.dim, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = self.dim
        self.layer1 = self._make_layer(self.dim,  stride=1)
        self.layer2 = self._make_layer(2*self.dim, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(2*self.dim, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class BasicEncoder4Evs(nn.Module):
    def __init__(self, bins=5, output_dim=128, dim=DIM, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder4Evs, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim
        self.bins = bins
        self.dim = dim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=self.dim)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(self.dim)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(self.dim)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(self.bins, self.dim, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = self.dim
        self.layer1 = self._make_layer(self.dim, stride=1)
        self.layer2 = self._make_layer(2*self.dim, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(2*self.dim, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.conv2(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)
    


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.padding = (kernel_size-1)//2
        # Depthwise Convolution (groups=in_channels ensures each channel is processed separately)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=self.padding, groups=in_channels, bias=False)
        
        # Pointwise Convolution (1x1 Conv to mix channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)  # Apply depthwise convolution
        x = self.pointwise(x)  # Apply pointwise convolution
        return x
    


class MKSmallEncoder(nn.Module):
    def __init__(self, in_channels, output_dim=128, norm_fn='batch', dropout=0.0):
        super(MKSmallEncoder, self).__init__()     #call the constructor of nn.Module

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.norm_fn = norm_fn

        #conv2d with 2 output channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        #depthwise separable convolution
        self.sep_conv1 = nn.Sequential(
            DepthwiseSeparableConv(2, 4, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        #here perform pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sep_conv2 = nn.Sequential(
            DepthwiseSeparableConv(4, 8, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.sep_conv3 = nn.Sequential(
            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sep_conv4 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Conv2d(32, 32, kernel_size=1)

        self.conv4 = nn.Conv2d(32, self.output_dim, kernel_size=1)

    def forward(self,x):
        b, n, c, h, w = x.shape
        x = x.view(b*n, c, h, w)

        x = self.conv1(x)
        x = self.sep_conv1(x)
        x = self.pool1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)
        x = self.pool2(x)
        x = self.sep_conv4(x)
        x = self.conv3(x)
        x = self.conv4(x)

        #return the output shape.
        # if output_dim = 128, MATCHING FEATURES
        # if output_dim = 384, CONTEXT FEATURES
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)



class MKBigEncoder(nn.Module):
    def __init__(self, in_channels, output_dim=128, norm_fn='batch', dropout=0.0):
        super(MKBigEncoder, self).__init__()     #call the constructor of nn.Module

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.norm_fn = norm_fn

        #conv2d with 2 output channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        #depthwise separable convolution
        self.sep_conv1 = nn.Sequential(
            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        #here perform pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sep_conv2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sep_conv3 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        

        self.sep_conv4 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Conv2d(128, self.output_dim, kernel_size=1)


    def forward(self,x):
        b, n, c, h, w = x.shape
        x = x.view(b*n, c, h, w)

        x = self.conv1(x)
        x = self.sep_conv1(x)
        x = self.pool1(x)
        x = self.sep_conv2(x)
        x = self.pool2(x)
        x = self.sep_conv3(x)
        x = self.sep_conv4(x)
        x = self.conv2(x)

        #return the output shape.
        # if output_dim = 128, MATCHING FEATURES
        # if output_dim = 384, CONTEXT FEATURES
        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)