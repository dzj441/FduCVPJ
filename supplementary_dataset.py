import numpy as np
from PIL import Image
import os
import shutil

'''
1. validate the new dataset
2. merge 2 sets
3. upload
'''

def check_image_files(dir1, dir2):
    # 获取第一个子目录中的所有.jpg文件
    jpg_files = [f for f in os.listdir(dir1) if f.endswith('.jpg')]
    png_files = [f for f in os.listdir(dir2) if f.endswith('.png')]
    print(len(jpg_files))
    print(len(png_files))

    # 遍历每个.jpg文件
    for jpg_file in jpg_files:
        # 获取同名的.png文件名
        png_file = os.path.splitext(jpg_file)[0] + '.png'
        if not png_file in png_files:
            print(f"没有找到同名的 .png 文件：{jpg_file}")
    print("all checked")


def move_files(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        print(f"源目录 {src_dir} 不存在！")
        return
    
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 获取源目录中的所有文件
    files = os.listdir(src_dir)
    
    for file in files:
        src_file = os.path.join(src_dir, file)
        dest_file = os.path.join(dest_dir, file)
            
        # 检查文件是否存在，如果是文件则进行移动
        if os.path.isfile(src_file):
            try:
                shutil.move(src_file, dest_file)
                print(f"已将 {file} 从 {src_dir} 移动到 {dest_dir}")
            except Exception as e:
                print(f"移动 {file} 时出错: {e}")
        else:
            print(f"{file} 不是文件，跳过。")
def generating_masks(raw_mask_dir,destination_dir):
    os.makedirs(destination_dir,exist_ok=True)

    for item in os.listdir(raw_mask_dir):
        im=Image.open(os.path.join(raw_mask_dir,item))
        im=(np.array(im).astype(float)/255*2).astype(int).astype('uint8')
        im = Image.fromarray(im)
        im.save(os.path.join(destination_dir,item))

if __name__ == '__main__':
    new_dataset_dir1 = r"data\refuge2\images"
    new_dataset_dir2 = r"data\refuge2\labels"
    # 1. check img files
    # check_image_files(new_dataset_dir1,new_dataset_dir2)
    
    # 2. merge dataset 
    # move_files(src_dir=r"data\refuge2\images",dest_dir=r"data\training\Disc_Cup_Mask")
    # move_files(src_dir=r"data\refuge2\labels",dest_dir=r"data\training\fundus_color_images")

    # 3. generate mask for paddle
    generating_masks(raw_mask_dir=r"data\training\Disc_Cup_Mask",destination_dir=r"data\training\mask")


