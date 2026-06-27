import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取一张图像（替换为自己的图像路径）
img = cv2.imread("ltt.png")   # OpenCV默认读取为BGR
# 将BGR格式转换为RGB格式（Matplotlib默认显示为RGB）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 显示图像
plt.imshow(img_rgb)
plt.axis("off")
plt.show()

print("环境搭建成功！")