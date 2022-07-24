"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""
import torch.nn as nn
import torch
# 基本卷积类
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
        self.relu   = nn.ReLU6(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6):
        super(LinearBottleNeck,self).__init__()

        self.residual = nn.Sequential(
            # 升维 
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            # Dwise
            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            # 降维
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            # 残差处理
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super(MobileNetV2,self).__init__()
        # 如果 input 224 * 224
        # conv2d    224 * 224 * 3             c = 32 n = 1 s = 2 
        self.pre = BasicConv2dBlock(3,32,1,padding = 1)
        # bottleneck 112 * 112 * 32    t = 1  c = 16  n = 1 s = 1
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        # bottleneck 112 * 112 * 16    t = 6  c = 24  n = 2 s = 2
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        # bottleneck 56  * 56  * 24    t = 6  c = 32  n = 3 s = 2
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        # bootleneck 28  * 28  * 32    t = 6  c = 64  n = 4 s = 2
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        # bootleneck 14  * 14  * 64    t = 6  c = 64  n = 3 s = 1
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        # bootleneck 14  * 14  * 96    t = 6  c = 160 n = 3 s = 2
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        # bootleneck 7  * 7   * 160   t = 6  c = 320 n = 1  s = 1
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)
        # bootleneck 7  * 7   * 320          c = 1280 n = 1 s = 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        # LinearBottleNeck 只有首次进行下采样
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


if __name__ == "__main__":
    pass