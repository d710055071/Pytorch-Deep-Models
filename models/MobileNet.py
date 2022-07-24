"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""


import torch.nn as nn
import torch
# 基本卷积类
class BasicConv2dBlock(nn.Module):
    # 构造方法
    def __init__(self,input_channels, output_channels, kernel_size,downsample = True, **kwargs):
        """基本卷积模块

        Args:
            input_channels        (int): 输入通道数
            output_channels       (int): 输出通道数
            kernel_size           (int): 卷积核大小
            downsample (bool, optional): 是否进行下采样(一些比较小的图片如果下采样后，到后面结构太小了). Defaults to True.
        """
        super(BasicConv2dBlock,self).__init__()
        # 判断是否进行下采样
        stride = 2 if downsample else 1
        # 卷积
        self.conv   = nn.Conv2d(input_channels,output_channels,kernel_size,stride=stride,**kwargs)
        # 批量归一化
        self.bn     = nn.BatchNorm2d(output_channels)
        # 激活函数
        self.relu   = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
# 深度可分离卷积
class DepthSeperabelConv2dBlock(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, **kwargs):
        super(DepthSeperabelConv2dBlock,self).__init__()
        # 深度卷积
        self.depth_wise = nn.Sequential(
            nn.Conv2d(input_channels,input_channels,kernel_size,groups=input_channels,**kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
        # 逐点卷积
        self.point_wise = nn.Sequential(
            nn.Conv2d(input_channels,output_channels,1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x
class MobileNet(nn.Module):
    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                        a network uniformly at each layer. For a given
                        layer and width multiplier α, the number of
                        input channels M becomes αM and the number of
                        output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
        super(MobileNet,self).__init__()

        alpha = width_multiplier

        self.stem = nn.Sequential(
            BasicConv2dBlock(3,int(32 * alpha),kernel_size = 3,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(32 * alpha),int(64 * alpha),kernel_size = 3,padding = 1,bias = False)
        )

        self.conv1 = nn.Sequential(
            DepthSeperabelConv2dBlock(int(64 * alpha),int(128 * alpha),kernel_size = 3,stride = 2,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(128 * alpha),int(128 * alpha),kernel_size = 3,padding = 1,bias = False)
        )

        self.conv2 = nn.Sequential(
            DepthSeperabelConv2dBlock(int(128 * alpha),int(256 * alpha),kernel_size = 3,stride = 2,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(256 * alpha),int(256 * alpha),kernel_size = 3,padding = 1,bias = False)
        )

        self.conv3 = nn.Sequential(
            DepthSeperabelConv2dBlock(int(256 * alpha),int(512 * alpha),kernel_size = 3,stride = 2,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(512 * alpha),int(512 * alpha),kernel_size = 3,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(512 * alpha),int(512 * alpha),kernel_size = 3,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(512 * alpha),int(512 * alpha),kernel_size = 3,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(512 * alpha),int(512 * alpha),kernel_size = 3,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(512 * alpha),int(512 * alpha),kernel_size = 3,padding = 1,bias = False)
        )
        self.conv4 = nn.Sequential(
            DepthSeperabelConv2dBlock(int(512 * alpha),int(1024 * alpha),kernel_size = 3,stride = 2,padding = 1,bias = False),
            DepthSeperabelConv2dBlock(int(1024 * alpha),int(1024 * alpha),kernel_size = 3,padding = 1,bias = False)
        )
        self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)

if __name__ == "__main__":
    net = mobilenet()
    dummy_input = torch.rand(1, 3, 244, 244)
    onnx_path = r"/home/dongzf/桌面/github/Pytorch-Deep-Models/mobile.onnx"
    torch.onnx.export(net, dummy_input, onnx_path,  input_names=['in'], output_names=['out']) 