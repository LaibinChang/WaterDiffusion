import os
import cv2
import numpy as np

# 输入和输出文件夹路径
input_folder = "E:\MaskOpera\OutputMAP"
output_folder = "E:\MaskOpera\OutR2Hot"

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 构建输入图像的完整路径
    input_image_path = os.path.join(input_folder, filename)

    # 读取图像
    original = cv2.imread(input_image_path)

    # 灰度化图像
    gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 计算灰度图像的最小值和最大值
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_img)

    # 将灰度图像归一化到 [0, 255] 的范围
    norm_img = cv2.normalize(gray_img, None, min_val, max_val, cv2.NORM_MINMAX)

    # 将归一化的图像转换为 uint8 类型
    norm_img = np.asarray(norm_img, dtype=np.uint8)

    # 使用 cv2.applyColorMap() 生成伪彩色图像（BGR 排列）
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

    # 将 BGR 转换为 RGB
    heat_img_rgb = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)

    # 构建输出图像的完整路径
    output_image_path = os.path.join(output_folder, filename)

    # 将叠加后的图像保存到输出文件夹中
    cv2.imwrite(output_image_path, heat_img_rgb)

print("图像处理完成！")