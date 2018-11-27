import torch.nn as nn
import torch.nn.functional as F

def conv_bn(input_channels, output_channels, kernel_size, stride, bias = False, padding = 0):
    """
    Introduction
    ------------
        基本卷积结构
    Parameters
    ----------
        input_channels: 输入通道数
        output_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 卷积步长
        bias: 是否使用偏置项
    """
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias),
        nn.BatchNorm2d(output_channels)
    )

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Introduction
        ------------
            FPN模型的bottleneck结构
        Parameters
        ----------
            input_channels: 输入图像通道数
            output_channels: 输出图像通道数
            expansion: 扩充比例
            stride: 卷积步长
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        """
        Introduction
        ------------
            特征图像金字塔结构(FPN)
        Parameters
        ----------
            block: bottle结构
            num_blocks: block数量
        """
        super(FPN, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride = 2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size = 3, stride = 2, padding = 1)
        self.relu2 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size = 3, stride = 2, padding = 1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size = 1, stride = 1, padding = 0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size = 1, stride = 1, padding = 0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size = 1, stride = 1, padding = 0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)

    def _upsample_add(self, x, y):
        """
        Introduction
        ------------
            特征融合，加上低层特征向上采样
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size = (H, W), mode = 'nearest') + y

    def _make_layer(self, block, output_channels, num_blocks, stride):
        """
        Introduction
        ------------
            构建ResNet网络结构
        Parameters
        ----------
            block: bottle网络结构
            output_channels: block输出通道数
            num_blocks: block的数量
            stride: block卷积步长
        """
        downsample = None
        if stride != 1 or self.inplanes != output_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, output_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, output_channels, stride, downsample))
        self.inplanes = output_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, output_channels))

        return nn.Sequential(*layers)


    def forward(self, input):
        c1 = F.relu(self.bn1(self.conv1(input)))
        c1 = F.max_pool2d(c1, kernel_size = 3, stride = 2, padding = 1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # 特征融合，高层特征上采样加上低层特征
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))

        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p5 = self.toplayer1(p5)
        p4 = self.toplayer1(p4)
        p3 = self.toplayer2(p3)
        return p3, p4, p5, p6, p7


def build_FPN_ResNet50():
    """
    Introduction
    ------------
        构建FPN ResNet50模型结构
    """
    return FPN(Bottleneck, [3, 4, 6, 3])
