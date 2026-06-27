from PIL import Image

# 读取图像
img = Image.open("ltt.png")
# 调整图像尺寸
img_resized = img.resize((640, 480))
# 转换为灰度图像
img_gray = img.convert("L")
# 保存图像
img_resized.save("resized_ltt.png")
# 显示图像
img_gray.show()
