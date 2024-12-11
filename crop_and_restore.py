import cv2
import numpy as np
import os
import json

# 裁剪图片
def crop_and_save(image_path, label_path, output_dir_image, output_dir_label, output_dir_meta, padding=20):
    # 创建输出目录
    os.makedirs(output_dir_image, exist_ok=True)
    os.makedirs(output_dir_label, exist_ok=True)
    os.makedirs(output_dir_meta, exist_ok=True)

    # 读取原始图像和标签图像
    image = cv2.imread(image_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # 标签图像为灰度图

    # 找到视杯和视盘的区域
    regions = np.where((label == 128))  # 获取视杯视盘的像素坐标
    if len(regions[0]) == 0 or len(regions[1]) == 0:
        print("未找到视杯或视盘区域，跳过裁剪")
        return

    # 计算最小和最大边界
    min_y, max_y = np.min(regions[0]), np.max(regions[0])
    min_x, max_x = np.min(regions[1]), np.max(regions[1])

    # 添加边界扩展，并限制范围
    height, width = label.shape
    min_y = max(0, min_y - padding)
    max_y = min(height, max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(width, max_x + padding)

    # 裁剪原始图像和标签图像
    cropped_image = image[min_y:max_y, min_x:max_x]
    cropped_label = label[min_y:max_y, min_x:max_x]

    # 保存裁剪结果
    base_name = os.path.basename(image_path).split('.')[0]
    cropped_image_path = os.path.join(output_dir_image, f"{base_name}.jpg")
    cropped_label_path = os.path.join(output_dir_label, f"{base_name}.png")
    cv2.imwrite(cropped_image_path, cropped_image)
    cv2.imwrite(cropped_label_path, cropped_label)

    # 保存裁剪元信息 and 转换为标准Python类型int64
    meta_data = {
        "original_width": int(width),
        "original_height": int(height),
        "min_y": int(min_y),
        "max_y": int(max_y),
        "min_x": int(min_x),
        "max_x": int(max_x)
    }
    meta_data_path = os.path.join(output_dir_meta, f"{base_name}.json")
    with open(meta_data_path, 'w') as f:
        json.dump(meta_data, f)

    print(f"裁剪完成，保存到：\n原始图像: {cropped_image_path}\n标签图像: {cropped_label_path}\n元信息: {meta_data_path}")

# 恢复到原始位置，来做test
def restore_image(cropped_image_path, meta_data_path, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    # 加载裁剪的图片和元信息
    cropped_image = cv2.imread(cropped_image_path)
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)
    # 创建一个与原始尺寸相同的白色画布（也就是其它区域补充成255）
    original_width = meta_data["original_width"]
    original_height = meta_data["original_height"]
    restored_image = np.full((original_height, original_width, 3), 255, dtype=np.uint8)
    # 获取裁剪区域坐标
    min_y = meta_data["min_y"]
    max_y = meta_data["max_y"]
    min_x = meta_data["min_x"]
    max_x = meta_data["max_x"]
    # 将裁剪图片放回原始位置
    restored_image[min_y:max_y, min_x:max_x] = cropped_image
    # 保存恢复后的图片
    base_name = os.path.basename(cropped_image_path).split('.')[0]
    restored_image_path = os.path.join(output_dir, f"{base_name}.png")
    cv2.imwrite(restored_image_path, restored_image)
    print(f"恢复完成，保存到：{restored_image_path}")


for i in range(1, 101):
    image_path = fr"D:\another_C\fundus_color_images\training\fundus_color_images\{i:04d}.jpg"
    label_path = fr"D:\another_C\Disc_Cup_Mask\training\Disc_Cup_Mask\{i:04d}.png"
    output_dir_image = r"D:\another_C\ccvv\fundus_color_images_ff"
    output_dir_label = r"D:\another_C\ccvv\Disc_Cup_Mask_ff"
    output_dir_meta = r"D:\another_C\ccvv\meta"
    restored_dir = r"D:\another_C\ccvv\restored_res"
    crop_and_save(image_path, label_path, output_dir_image, output_dir_label, output_dir_meta)

    cropped_image_path = os.path.join(output_dir_label, f"{i:04d}.png")
    meta_data_path = os.path.join(output_dir_meta, f"{i:04d}.json")
    restore_image(cropped_image_path, meta_data_path, restored_dir)
