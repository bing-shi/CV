import torch
import torch.nn as nn
import torch.optim as optim
from e3_1 import SimpleCNN

# 1. 初始化核心组件
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)       # 模型移至GPU/CPU，实现加速
criterion = nn.CrossEntropyLoss()    # 分类任务损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 首选Adam优化器

# 2. 模拟视觉训练数据（批量大小4，3通道224×224标准图像）
train_data = torch.randn(4, 3, 224, 224).to(device)
train_labels = torch.tensor([0,1,2,0]).to(device)     # 对应图像分类标签

# 3. 完整训练循环
epochs = 10
for epoch in range(epochs):
    model.train()                     # 切换训练模式，启用Dropout/BatchNorm

    # 前向传播：图像输入模型，得到预测结果
    outputs = model(train_data)
    loss = criterion(outputs, train_labels)  # 计算预测与真实标签误差

    # 反向传播+参数更新（核心步骤）
    optimizer.zero_grad()            # 清空上一轮梯度，必做！避免梯度累积导致不收敛
    loss.backward()                   # 自动求导，计算所有参数梯度
    optimizer.step()                  # 根据梯度更新网络参数

    # 打印训练信息，监控训练状态
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")