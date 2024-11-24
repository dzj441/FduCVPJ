import os
from PIL import Image
'''
convert the bmp result to png
'''
def convert_bmp_to_png_in_directory(input_dir, output_dir):
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.bmp'):
            # 构建完整的文件路径
            bmp_path = os.path.join(input_dir, filename)
            
            # 构建 PNG 文件路径（保持文件名，但扩展名为 .png）
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(output_dir, png_filename)

            try:
                # 打开 BMP 图像并保存为 PNG
                with Image.open(bmp_path) as img:
                    img.save(png_path, format="PNG")
                print(f"Converted {bmp_path} to {png_path}")
            except Exception as e:
                print(f"Error converting {bmp_path}: {e}")

# 示例：
input_directory = "data\Training_Disc_Cup_Segmentations"  # 输入 BMP 图像文件夹路径
output_directory = "data\predict"  # 输出 PNG 图像文件夹路径

convert_bmp_to_png_in_directory(input_directory, output_directory)
