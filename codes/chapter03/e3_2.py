#  nn.Seuqential构建顺序深度视觉网络
import torch
import torch.nn as nn

# 构建深度卷积网络（序贯方式，三层卷积块+分类头）
DeepCNN = nn.Sequential(
    # 第1卷积块：浅层特征提取+下采样
    nn.Conv2d(3, 32, 3, 1, 1),
    nn.BatchNorm2d(32),                # 批归一化，解决内部协变量偏移，加速训练
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # 第2卷积块：中层特征提取+下采样
    nn.Conv2d(32, 64, 3, 1, 1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # 第3卷积块：深层高级特征提取+下采样
    nn.Conv2d(64, 128, 3, 1, 1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    # 展平+全连接分类头，nn.Flatten替代view()，代码更简洁
    nn.Flatten(),
    nn.Linear(128*28*28, 256),
    nn.ReLU(),
    nn.Dropout(0.5),                    # Dropout随机失活神经元，防止过拟合
    nn.Linear(256, 10)
)

# 测试网络输出，验证结构正确性
test_input = torch.randn(2, 3, 224, 224)
output = DeepCNN(test_input)
print("深度网络输出：", output.shape)  # 输出2×10，适配10分类