import cv2
import numpy as np
import os
from scipy import ndimage
from tqdm import tqdm
import pandas as pd

class PostProcessor:
    def __init__(self, kernel_size=(5, 5), sigma=0, threshold=127):
        """
        初始化后处理器
        :param kernel_size: 高斯平滑核大小
        :param sigma: 高斯平滑标准差
        :param threshold: 二值化阈值
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.threshold = threshold

    def process_single(self, image_path):
        """
        处理单张图像
        :param image_path: 图像路径
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 高斯平滑
        smoothed = cv2.GaussianBlur(image, self.kernel_size, self.sigma)
        _, binary = cv2.threshold(smoothed, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 连通域分析
        labeled_array, num_features = ndimage.label(binary)
        if num_features == 0:
            return binary
            
        # 保留最大连通域
        sizes = ndimage.sum(binary, labeled_array, range(1, num_features + 1))
        max_label = np.argmax(sizes) + 1
        final_mask = (labeled_array == max_label).astype(np.uint8) * 255
        
        return final_mask

    def process_batch(self, input_folder):
        """
        处理整个文件夹中的图像
        :param input_folder: 输入文件夹路径
        :return: 处理后的输出文件夹路径:/input_folder/post_processed
        """
        output_folder = os.path.join(input_folder, 'post_processed')
        os.makedirs(output_folder, exist_ok=True)
        
        for filename in tqdm(os.listdir(input_folder)):
            if filename.endswith('.png'):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                # 读取图像
                image = input_path
                if image is None:
                    continue
                    
                # 处理并保存
                processed = self.process_single(image)
                cv2.imwrite(output_path, processed)
                
        return output_folder
    
class SubmitProcessor:
    def __init__(self, output_path):
        """
        初始化提交处理器
        :param output_path: 最终输出路径
        """
        self.output_path = output_path
        self.submit_folder = os.path.join(output_path, 'Segmentation_Results')
        os.makedirs(self.submit_folder, exist_ok=True)

    def process(self, mask_folder, cls_xlsx):
        """
        处理分割结果和分类结果
        :param mask_folder: 后处理后的掩码文件夹路径
        :param cls_xlsx: 分类结果Excel文件路径
        """
        # 读取分类结果
        try:
            cls_df = pd.read_excel(cls_xlsx)
            cls_dict = dict(zip(cls_df['Image'], cls_df['Pterygium']))
        except Exception as e:
            print(f"读取分类结果文件出错: {e}")
            return
        
        # 处理每个掩码文件
        mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
        for mask_file in tqdm(mask_files, desc="处理提交结果"):
            try:
                # 获取图像编号：0451_mask.png -> 0451
                img_num = int(os.path.splitext(mask_file)[0].split('_')[0])
                # 获取分类结果
                cls_result = cls_dict.get(img_num, -1)
                if cls_result == -1:
                    print(f"警告: 未找到图像 {img_num} 的分类结果")
                    continue
                
                # 读取掩码
                mask_path = os.path.join(mask_folder, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # 创建三通道结果图像
                result = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                
                # 根据分类结果处理
                if cls_result != 0:  # 非正常样本
                    # 将掩码中的255转换为128，并只放在R通道
                    result[:, :, 2] = (mask > 0).astype(np.uint8) * 128
                
                # 保存结果
                output_path = os.path.join(self.submit_folder, f"{img_num:04d}.png")
                cv2.imwrite(output_path, result)
                
            except Exception as e:
                print(f"处理图像 {mask_file} 时出错: {e}")
                continue
        
        print(f"处理完成，结果保存在: {self.submit_folder}")
