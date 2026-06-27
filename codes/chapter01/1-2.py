import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定Windows内置宋体的字体文件
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 宋体（simsun）
# 或微软雅黑：C:/Windows/Fonts/msyh.ttc
font_prop = font_manager.FontProperties(fname=font_path)

# 读取灰度图像（可替换为自己的图像路径）
img = cv2.imread("ltt.png", cv2.IMREAD_GRAYSCALE)
# 为图像添加高斯噪声（模拟实际含噪图像）
mean = 0
var = 0.001
sigma = var**0.5
noise = np.random.normal(mean, sigma, img.shape)
noise = noise.astype(np.uint8)
noisy_img = cv2.add(img, noise)

# 分别进行均值滤波、中值滤波、高斯滤波、双边滤波
mean_blur = cv2.blur(noisy_img, (3,3))
median_blur = cv2.medianBlur(noisy_img, 3)
gaussian_blur = cv2.GaussianBlur(noisy_img, (3,3), 0)
bilateral_blur = cv2.bilateralFilter(noisy_img, 9, 75, 75)

# 显示对比结果
plt.figure(figsize=(12, 8), num="Lenna")
plt.subplot(2,3,1)
plt.imshow(img, cmap="gray")
plt.title("原始图像", fontproperties=font_prop)
plt.axis("off")
plt.subplot(2,3,2)
plt.imshow(noisy_img, cmap="gray")
plt.title("噪声图像", fontproperties=font_prop)
plt.axis("off")
plt.subplot(2,3,3)
plt.imshow(mean_blur, cmap="gray")
plt.title("均值滤波", fontproperties=font_prop)
plt.axis("off")
plt.subplot(2,3,4)
plt.imshow(median_blur, cmap="gray")
plt.title("中值滤波", fontproperties=font_prop)
plt.axis("off")
plt.subplot(2,3,5)
plt.imshow(gaussian_blur, cmap="gray")
plt.title("高斯滤波", fontproperties=font_prop)
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(bilateral_blur, cmap="gray")
plt.title("双边滤波", fontproperties=font_prop)
plt.axis('off')
plt.tight_layout()
plt.show()

# 计算滤波后图像的信噪比
def calculate_snr(original, filtered):
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)
    noise = original - filtered
    snr=10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr

print(f"均值滤波SNR: {calculate_snr(img, mean_blur):.2f}")
print(f"中值滤波SNR: {calculate_snr(img, median_blur):.2f}")
print(f"高斯滤波SNR: {calculate_snr(img, gaussian_blur):.2f}")
print(f"双边滤波SNR: {calculate_snr(img, bilateral_blur):.2f}")
