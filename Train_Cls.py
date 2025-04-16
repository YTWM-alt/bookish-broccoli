import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights,convnext_base,ConvNeXt_Base_Weights
import random
import setproctitle
import warnings
warnings.filterwarnings("ignore")
setproctitle.setproctitle("ZGB_预计用至4/16/13:00")
# setproctitle.setproctitle("ZGB_代码调试_不长时间运行")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置要使用的GPU的索引

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 如果使用 GPU，设置 CUDA 随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 设置所有可用 GPU 的种子
        # 确保在每次前向传播时使用相同的随机性（如果使用了 dropout）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PTEClassificationDataset(IterableDataset):
    def __init__(self, root_dir, xlsx_path, phase='train', patient_ids=None, 
                 shuffle=True, seed=42):
        """
        初始化数据集
        :param root_dir: 图像根目录
        :param xlsx_path: 标签文件路径
        :param phase: 训练/验证阶段
        :param patient_ids: 指定的病人ID列表
        :param shuffle: 是否打乱数据
        :param seed: 随机种子
        """
        self.root_dir = root_dir
        self.phase = phase
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        
        # 读取Excel文件
        self.df = pd.read_excel(xlsx_path)
        
        # 根据patient_ids筛选数据
        if patient_ids:
            self.df = self.df[self.df['Image'].isin(patient_ids)]
            
        # 基础预处理
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),  
            transforms.ToTensor(),
            # 理论上应该是需要使用预训练模型的均值和标准差，但实验表明使用本数据集的均值和标准差收敛更快
            # 不过使用预训练模型的均值和标准差应该也不会有太大问题，还是和其他工作保持一致
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.Normalize(mean=[0.6151, 0.4225, 0.3555], std=[0.2180, 0.2214, 0.2201])
        ])
    
    def __iter__(self):
        """实现迭代式数据加载"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 准备数据
        if worker_info is None:  # 单worker
            df_split = self.df
        else:  # 多worker分片
            per_worker = int(np.ceil(len(self.df) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.df))
            df_split = self.df.iloc[start:end]
        
        # 转换为列表并可选打乱
        items = list(df_split.itertuples())
        if self.shuffle:
            self.rng.shuffle(items)
        
        # 流式生成样本
        for item in items:
            img_path = os.path.join(self.root_dir, self.phase, f"{item.Image:04d}", f"{item.Image:04d}.png")
            try:
                # 加载和预处理图像
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                label = item.Pterygium  # 获取分类标签
                
                yield img_tensor, label
                
            except Exception as e:
                print(f"处理图像出错 {img_path}: {e}")
                continue

class ConvNextClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ConvNextClassifier, self).__init__()
        # 加载预训练的ConvNext-Base
        # self.convnext = timm.create_model("hf_hub:timm/convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True)
        # self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        self.conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.conv.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(1), nn.LayerNorm(768), nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def main():
    # 设置随机种子和设备
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("Training on", n_gpu, "GPUs.")
    else:
        n_gpu = 0
        print("Training on CPU.")
    
    # 创建数据集
    train_dataset = PTEClassificationDataset(
        root_dir=r'/home/data/zgb/datasets/pterygium_on_the_eyeball',
        xlsx_path=r'/home/data/zgb/datasets/pterygium_on_the_eyeball/train/train_classification_label.xlsx',
        phase='train',
        patient_ids=[i for i in range(1, 361)],
        shuffle=True
    )
    
    val_dataset = PTEClassificationDataset(
        root_dir=r'/home/data/zgb/datasets/pterygium_on_the_eyeball',
        xlsx_path=r'/home/data/zgb/datasets/pterygium_on_the_eyeball/train/train_classification_label.xlsx',
        phase='train',
        patient_ids=[i for i in range(361, 450)],
        shuffle=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=30,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=30,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = ConvNextClassifier(num_classes=3)
    model = nn.DataParallel(model).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)
    
    # 训练参数
    num_epochs = 60
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    scaler = torch.amp.GradScaler()
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        correct = 0
        total = 0   
        count_train = 0     
        for inputs, labels in train_loader:
            count_train += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        train_loss = train_loss / count_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        count_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                count_val += 1
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / count_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'classifier_best.pth')
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            print("Early stopping...")
            break
    # 保存最后训练的模型
    torch.save(model.state_dict(), 'classifier_last.pth')
    
    # 保存训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('classification_curves.png')
    plt.close()

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()