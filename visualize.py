import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_comparison(predicted_img, ground_truth_img, save_path, file_name):
    """
    可视化预测图像和真实图像的左右对比，同时标识它们的来源。
    """
    # 创建一个 1x2 的子图布局（左边显示 GT，右边显示预测）
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    
    # 显示 Ground Truth 图像 (左边)
    axes[0].imshow(ground_truth_img)
    axes[0].axis('off')  # 关闭坐标轴
    axes[0].set_title("Ground Truth", fontsize=15)
    
    # 显示预测图像 (右边)
    axes[1].imshow(predicted_img)
    axes[1].axis('off')  # 关闭坐标轴
    axes[1].set_title("Predicted", fontsize=15)

    # 设置图像标题
    fig.suptitle(f"Comparison for {file_name}", fontsize=20)

    # 保存结果图像
    save_file = os.path.join(save_path, f"{file_name}.png")
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Saved comparison image to {save_file}")

# 在脚本直接运行时执行以下代码
if __name__ == '__main__':
    predictedPath = "data/predict"  # 预测图像文件夹
    groundTruthPath = "data/training/Disc_Cup_Mask"  # Ground Truth 图像文件夹
    savePath = "data/maskComparison"  # 输出结果保存路径

    # 确保输出目录存在
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    predicted_files = sorted(os.listdir(predictedPath))
    ground_truth_files = sorted(os.listdir(groundTruthPath))

    # 遍历文件列表并处理图像
    for predicted_file, gt_file in zip(predicted_files, ground_truth_files):
        if predicted_file.endswith('.png') and gt_file.endswith('.png'):
            # 构建完整路径
            predicted_image_path = os.path.join(predictedPath, predicted_file)
            ground_truth_image_path = os.path.join(groundTruthPath, gt_file)
            
            # 加载图片
            predicted_img = plt.imread(predicted_image_path)
            ground_truth_img = plt.imread(ground_truth_image_path)

            # 提取文件名（不包含扩展名）
            file_name = os.path.splitext(predicted_file)[0]

            # 绘制并保存对比图像
            visualize_comparison(predicted_img, ground_truth_img, savePath, file_name)
