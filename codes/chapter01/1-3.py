import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定Windows内置宋体的字体文件
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 宋体（simsun）
# 或微软雅黑：C:/Windows/Fonts/msyh.ttc
font_prop = font_manager.FontProperties(fname=font_path)

# 读取灰度图像
img = cv2.imread("ltt.png", cv2.IMREAD_GRAYSCALE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # 用于绘制彩色角点

# Harris角点检测
dst = cv2.cornerHarris(img, 2, 3, 0.04)
# 膨胀角点，使角点更明显（便于可视化）
dst = cv2.dilate(dst, None)
# 设定阈值，标记角点（红色）
img_color[dst > 0.01 * dst.max()] = [0, 0, 255]
# Canny边缘检测
canny_edges = cv2.Canny(img, 100, 200)

# 显示结果
plt.figure(figsize=(12, 8), num="Lenna")
plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("原始图像", fontproperties=font_prop)
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Harris角点检测", fontproperties=font_prop)
plt.axis("off")
plt.subplot(1,3,3)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny边缘检测", fontproperties=font_prop)
plt.axis("off")
plt.tight_layout()
plt.show()