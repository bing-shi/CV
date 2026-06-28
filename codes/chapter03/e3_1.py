## PyTorch 手动构建卷积神经网络
import torch
import torch.nn as nn

def init_weights(m):
    """自定义初始化函数，适配视觉卷积网络"""
    if isinstance(m, nn.Conv2d):        # 卷积层He初始化，适配ReLU激活函数
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):      # 全连接层正态分布初始化
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):     # 层定义（同3.5.1）
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*56*56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """前向传播（x：批量图像张量，shape=[batch, 3, 224, 224]）"""
        # 卷积+池化（视觉浅层+深层特征提取）
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # 展平（特征图→一维特征向量，适配全连接层输入）
        x = x.view(x.size(0), -1)  # batch×(32×56×56)
        # 全连接层完成分类
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # 测试前向传播，验证网络结构是否正常
    model = SimpleCNN()
    model.apply(init_weights)
    test_input = torch.randn(2, 3, 224, 224)     # 2张测试图像，批量大小为2
    output = model(test_input)
    print("输出形状：", output.shape)                # 输出2×10，对应批量2，10分类任务