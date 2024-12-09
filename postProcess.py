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

- 对于模型预测分割的视杯视盘几乎为空做特殊处理，以已有的预测区域的中心位置作为参照，人为加上视盘视杯，将模型预测得到的零星位置信息充分利用以减小模型预测失误带来的损失
    - 进一步需要调整判断是否需要特殊处理的条件

- 对于完全空的预测结果用canny分得的大概轮廓取中心作为同心圆心


'''
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

import matplotlib.pyplot as plt

def split_areas(img):
    OD = np.zeros_like(img)
    OC = np.zeros_like(img)
    OD[img < 200] = 255
    OC[img < 10] = 255
    
    
    return OD, OC

# 去除小的联通区，只保留最大的
def save_max_area(Oimg, img_name):
    contours, hierarchy = cv2.findContours(Oimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return Oimg, 0
    area = [cv2.contourArea(contour) for contour in contours]
    max_idx = np.argmax(area)
    max_contour_area = area[max_idx]
    
    for i in range(len(contours)):
        if i != max_idx:
            cv2.fillPoly(Oimg, [contours[i]], 0)

    return Oimg, max_contour_area


def fit_and_fill_ellipse(contoured_img):
    contours, _ = cv2.findContours(contoured_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return contoured_img  

    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    if len(max_contour) >= 5:
        ellipse = cv2.fitEllipse(max_contour)
        
        ellipse_mask = np.zeros_like(contoured_img)
        
        cv2.ellipse(ellipse_mask, ellipse, (255, 255, 255), -1)
        
        # # 将掩模应用到原始图像上，只保留椭圆内部的区域
        # contoured_img = cv2.bitwise_and(contoured_img, ellipse_mask)
        contoured_img = ellipse_mask
    else:
        print("Not enough points to fit an ellipse.")

    return contoured_img


## 填充孔洞（闭运算）
def fill_hole(Oimg, k_size=300):
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


def fill_almost_empty_pred(OD, OC):
    # 找到 OD 联通区域的中心点
    contours, _ = cv2.findContours(OD, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        rD = 150
        rC = 80
        # 在 OD 中绘制圆
        cv2.circle(OD, (cX - (int)(rD/1.4), cY - (int)(rD/1.4)), rD, (255, 255, 255), -1) 

        # 在 OC 中绘制较小半径的圆
        cv2.circle(OC, (cX - (int)(rD/1.4), cY - (int)(rD/1.4)), rC, (255, 255, 255), -1)  

        plt.imshow(OD, cmap='gray')
        # plt.title(img_name)
        plt.show()

        plt.imshow(OC, cmap='gray')
        # plt.title(img_name)
        plt.show()
    
    return OD, OC 

def fill_totally_empty_pred(img, OD, OC):
    plt.imshow(img)
    plt.show()

    edges = cv2.Canny(img, threshold1=100, threshold2=200)

    plt.imshow(edges, cmap='gray')
    plt.show()
    
    _, binary_img = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    
    height, width = binary_img.shape
    start_x = width // 4
    end_x = 3 * width // 4
    start_y = height // 3
    end_y = 2 * height // 3
    
    cropped_img = binary_img[start_y:end_y, start_x:end_x]

    plt.imshow(cropped_img, cmap='gray')
    plt.show()
    
    # 识别白色点
    white_points = np.column_stack(np.where(cropped_img == 255)).transpose()
    
    if white_points.size == 0:
        return None, None
    
    center = np.mean(white_points, axis=0).astype(int)
    
    original_center = (center[1] + start_x, center[0] + start_y)


    print(f"Center in original image: {original_center}")
    cX, cY = original_center
    rD = 150
    rC = 80
    cv2.circle(OD, (cX, cY), rD, (255, 255, 255), -1)
    cv2.circle(OC, (cX, cY), rC, (255, 255, 255), -1)

    plt.imshow(OD, cmap='gray')
    # plt.title(img_name)
    plt.show()

    plt.imshow(OC, cmap='gray')
    plt.show()
    
    return OD, OC





def one_package_service(img, original_img, k_size=300, img_name=None, threshold=1500):
    OD, OC = split_areas(img)
    OD, OD_area = save_max_area(OD, img_name)
    OC, OC_area = save_max_area(OC, img_name)
   


    if OD_area <= threshold and OC_area <= threshold:
        print("fill_empty_pred")
        if OD_area == 0 and OC_area == 0:
            OD, OC = fill_totally_empty_pred(original_img, OD, OC)
        else:
            OD, OC = fill_almost_empty_pred(OD, OC)
    
    else:
        # pass
        OD = fill_hole(OD, k_size)
        OD = fit_and_fill_ellipse(OD)

        
        OC = fill_hole(OC, k_size)
        OC = fit_and_fill_ellipse(OC)


    result = merge_areas(OD, OC)
    return result

def process(input_folder='Disc_Cup_Segmentations', output_folder='Processed_Images', original_img_folder='testing/fundus color images', k_size=300, threshold=1500):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    imgs_name = os.listdir(input_folder)
    for name in tqdm(imgs_name):
        img_path = os.path.join(input_folder, name)
        original_img_path = os.path.join(original_img_folder, name.split('.')[0]+'.jpg')

        img = np.asarray(Image.open(img_path))
        original_img = np.asarray(Image.open(original_img_path))

        result = Image.fromarray(one_package_service(img, original_img, k_size=k_size, img_name=name, threshold=threshold))
        result.save(img_path.replace(input_folder, output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    imgs_name = os.listdir(input_folder)
    for name in tqdm(imgs_name):
        img_path = os.path.join(input_folder, name)
        img = np.asarray(Image.open(img_path))
        result = Image.fromarray(one_package_service(img, k_size=k_size, img_name=name))
        result.save(img_path.replace(input_folder, output_folder))