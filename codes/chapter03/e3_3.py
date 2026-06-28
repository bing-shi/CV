#带命名的 nn.Seuqential
import torch
import torch.nn as nn

# 给层命名（视觉网络调试、特征提取）
DeepCNN_named = nn.ModuleDict({
        "conv_block1": nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ),
        "conv_block2": nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        ),
        "classifier": nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*56*56, 10)
        )
    })

# 访问指定卷积块，提取中间视觉特征
test_input = torch.randn(2, 3, 224, 224)
feat = DeepCNN_named['conv_block2'](DeepCNN_named['conv_block1'](test_input))
print("中间特征形状：", feat.shape)                    # 输出2×64×56×56，中层特征图尺寸

output = DeepCNN_named['classifier'](feat)
print("深度网络输出：", output.shape)              # 输出2×10，适配10分类
print("Linear原始输出(logits):\n", output)

# 手动转为概率
prob = torch.softmax(output, dim=1)
print("Softmax后类别概率:\n", prob)
print("每行求和校验：", prob.sum(dim=1))       # 近似等于1

# 注意：因为网络没有经过训练，所以推理的结果是没有意义的、随机值。