'''

将预测结果的小的连通区域去除，只保留最大联通区域， 并且将其中的空洞部分填充，目前k_size=300能将测试结果从7.86->7.878，后面再尝试继续调整k_size，和添加椭圆拟合

用法说明：
- 导入这个文件中的process函数即可
```python
from postProcess import process
```
- 调用时可以选择预测结果的路径和存储后处理结果的路径，也可以默认, 会直接将后处理的结果存储在路径中
```python
def process(input_folder='Disc_Cup_Segmentations', output_folder='Processed_Images', k_size=300)
```
- k_size参数相当于画笔的粗细，越大画笔越粗，能够填补的空洞越大，但是细节丢失越多


'''

import numpy as np
import cv2
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


## 分离两个区域
def split_areas(img):
    OD = np.zeros_like(img)
    OC = np.zeros_like(img)
    OD[img < 200] = 255
    OC[img < 10] = 255
    return OD, OC


## 去除小的联通区，只保留最大的
def save_max_area(Oimg):
    contours, hierarchy = cv2.findContours(Oimg, \
    										  cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    if not contours:  # 检查contours是否为空
        return Oimg  # 如果为空，直接返回原始图像                                          
    area = []  # 找到最大区域并填充
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    # max_area = cv2.contourArea(contours[max_idx])
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(Oimg, [contours[i]], 0)
    return Oimg


## 填充孔洞（闭运算）
def fill_hole(Oimg, k_size=50):
    kernel = np.ones((k_size, k_size), np.uint8)
    close_img = cv2.morphologyEx(Oimg, cv2.MORPH_CLOSE, kernel)
    return close_img


## 合并区域为结果
def merge_areas(OD, OC):
    result = OD.copy()
    result = 255 - result
    result[result == 0] = 128
    result[OC == 255] = 0
    return result


## 一条龙服务
def one_package_service(img, k_size=50):
    OD, OC = split_areas(img)
    OD = save_max_area(OD)
    OD = fill_hole(OD, k_size)
    OC = save_max_area(OC)
    OC = fill_hole(OC, k_size)
    result = merge_areas(OD, OC)
    return result

def process(input_folder='Disc_Cup_Segmentations', output_folder='Processed_Images', k_size=300):

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    imgs_name = os.listdir(input_folder)
    for name in tqdm(imgs_name):
        img_path = os.path.join(input_folder, name)
        # print(img_path)
        img = np.asarray(Image.open(img_path))
        result = Image.fromarray(one_package_service(img, k_size=k_size))
        result.save(img_path.replace(input_folder, output_folder))