import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from Train_Cls import ConvNextClassifier  # 导入模型定义
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ClassificationInferencer:
    def __init__(self, model_path, device='cuda'):
        """
        初始化分类推理器
        :param model_path: 模型权重文件路径
        :param device: 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() and device=='cuda' else 'cpu')
        print(f"Using device: {self.device}")

        # 初始化模型
        self.model = ConvNextClassifier(num_classes=3)
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)
    
    @torch.no_grad()
    def predict_single(self, image_path):
        """
        对单张图片进行推理
        :param image_path: 输入图片路径
        :return: 预测类别和对应的概率
        """
        # 预处理
        image = self.preprocess_image(image_path)
        image = image.to(self.device)
        
        # 推理
        outputs = self.model(image)
        probabilities = torch.softmax(outputs, dim=1)
        
        # 获取预测结果
        pred_class = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0][pred_class].item()
        
        return pred_class, pred_prob
    
    @torch.no_grad()
    def predict_batch(self, input_dir, output_xlsx):
        """
        对文件夹中的所有图片进行批量推理
        :param input_dir: 输入图片文件夹路径
        :param output_xlsx: 输出Excel文件路径
        """
        results = []
        # 获取所有图片文件
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc="Processing images"):
            try:
                image_number = int(os.path.splitext(image_file)[0])
                image_path = os.path.join(input_dir, image_file)
                pred_class, pred_prob = self.predict_single(image_path)
                # 保存结果
                results.append({
                    'Image': image_number,
                    'Pterygium': pred_class,
                    'Probability': pred_prob
                })
                
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # 将结果保存为Excel文件
        df = pd.DataFrame(results)
        df = df.sort_values('Image')  # 按图像编号排序
        df.to_excel(output_xlsx, index=False)
        print(f"Results saved to {output_xlsx}")

def main():
    """使用示例"""
    # 初始化推理器
    inferencer = ClassificationInferencer(model_path='classifier_best.pth',device='cuda')
    # 单张图片推理示例
    image_path = 'val/0451.png'
    pred_class, pred_prob = inferencer.predict_single(image_path)
    print(f"Single image prediction:")
    print(f"Class: {pred_class}, Probability: {pred_prob:.4f}")
    
    # 批量推理示例
    input_dir = 'val'
    output_xlsx = 'prediction_results.xlsx'
    inferencer.predict_batch(input_dir, output_xlsx)

if __name__ == '__main__':
    main()