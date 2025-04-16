import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
from egeunet import EGEUNet  # 确保模型文件在同一目录下
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Inferencer:
    def __init__(self, model_path, device, conversion='RGB'):
        """
        初始化推理器
        :param model_path: 模型权重文件路径
        :param device: 使用的设备 ('cuda' 或 'cpu')
        :param conversion: 转换模式 ('RGB' 或 'L')
        """
        if device == 'cuda':
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu') 
        else:
            self.device = torch.device('cpu')

        # 初始化模型
        self.conversion = conversion
        if self.conversion == 'RGB':
            self.model = EGEUNet(1, 3, bridge=True, gt_ds=True)
        elif self.conversion == 'L':
            self.model = EGEUNet(1, 1, bridge=True, gt_ds=True)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 定义图像预处理
        self.transform_RGB = transforms.Compose([
            transforms.Resize((3456, 5184)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6151, 0.4225, 0.3555], 
                std=[0.2180, 0.2214, 0.2201]
            )
        ])
        self.transform_Gray = transforms.Compose([
            transforms.Resize((3456, 5184)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4678], std=[0.2182]) 
        ])
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        image = Image.open(image_path).convert(self.conversion)
        if self.conversion == 'RGB':
            return self.transform_RGB(image).unsqueeze(0) 
        elif self.conversion == 'L':
            return self.transform_Gray(image).unsqueeze(0) 
    
    def postprocess_output(self, output):
        """后处理模型输出"""
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        return output
    
    @torch.no_grad()
    def predict_single(self, image_path, save_path=None):
        """
        对单张图片进行推理
        :param image_path: 输入图片路径
        :param save_path: 保存结果的路径
        :return: 预测的掩码
        """
        # 预处理
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # 推理
        output = self.model(image)[-1]  # 取最后一个输出
        mask = self.postprocess_output(output)
        
        # 转换为numpy数组
        mask = mask.cpu().numpy()[0, 0]  # (H, W)
        
        # 保存结果
        if save_path:
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_image.save(save_path)
        
        return mask
    
    def predict_batch(self, input_dir, output_dir):
        """
        对文件夹中的所有图片进行批量推理
        :param input_dir: 输入图片文件夹路径
        :param output_dir: 输出结果保存文件夹路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        # 批量处理
        for image_file in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_mask.png")
            self.predict_single(input_path, output_path)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Model Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model weights file')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the output')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = Inferencer(args.model_path, args.device)
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input_path):
        # 单张图片处理
        inferencer.predict_single(args.input_path, args.output_path)
        print(f"Processed single image: {args.input_path}")
    elif os.path.isdir(args.input_path):
        # 批量处理目录
        inferencer.predict_batch(args.input_path, args.output_path)
        print(f"Processed directory: {args.input_path}")
    else:
        print("Error: Input path does not exist!")

if __name__ == '__main__':
    main()
    # 命令行使用方法
    # 处理单张图片
    # python inference.py --model_path UNet_best.pth --input_path path/to/image.png --output_path path/to/output.png
    # 处理整个文件夹
    # python inference.py --model_path UNet_best.pth --input_path path/to/input_folder --output_path path/to/output_folder
    