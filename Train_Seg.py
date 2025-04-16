from PIL import Image, ImageOps
from tqdm import tqdm
from egeunet import EGEUNet
from torchvision.transforms import functional as TF
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import IterableDataset, Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore")
import setproctitle
setproctitle.setproctitle("ZGB_预计用至4/15/4:00")
# setproctitle.setproctitle("ZGB_代码调试_不长时间运行")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置要使用的GPU的索引

class ComboLoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0, 
                 focal_weight=0.0,smooth=1e-6,gamma=2.0,alpha=0.25):
        
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha
        
        # 基础损失函数
        self.ce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        输入:
            inputs: 模型输出的logits (未经sigmoid), shape [B, 1, H, W]
            targets: 真实标签, shape [B, 1, H, W] (值应为0或1)
        """
        # 确保数据格式正确
        assert inputs.shape == targets.shape, f"输入形状不匹配: {inputs.shape} vs {targets.shape}"
        
        # 初始化总损失
        total_loss = 0.0
        
        # 计算交叉熵损失
        if self.ce_weight > 0:
            ce_loss = self.ce(inputs, targets)
            total_loss += self.ce_weight * ce_loss
        
        # 计算Dice损失
        if self.dice_weight > 0:
            # 应用sigmoid得到概率图
            probs = torch.sigmoid(inputs)
            
            # 计算交集和并集
            intersection = (probs * targets).sum(dim=(1,2,3))
            union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
            
            # Dice系数
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice.mean()
            
            total_loss += self.dice_weight * dice_loss
        
        # 计算Focal Loss
        if self.focal_weight > 0:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
            # Focal Loss计算
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            focal_loss = focal_loss.mean()
            
            total_loss += self.focal_weight * focal_loss
            
        return total_loss

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1, alpha=0.6, beta=0.4, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):

        # 模型中包含 sigmoid 或等效的激活层
        # inputs = F.softmax(inputs)

        # 将标签和预测张量展平
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算真正例、假正例和假负例
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        # 计算 Tversky 系数
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        # 计算 FocalTversky 损失
        FocalTversky = (1 - Tversky)**self.gamma

        return FocalTversky

def dice_coeff(pred, target, epsilon=1e-5):
    """
    计算Dice系数
    :param pred: 模型预测输出，维度为(N, C, H, W),类型为torch.Tensor
    :param target: 真实标签，维度为(N, C, H, W),类型为torch.Tensor，与pred形状相同
    :param epsilon: 避免除零错误的小值
    :return: Dice系数
    """
    # 将预测值和目标值转换为概率图
    pred = torch.sigmoid(pred)

    # 计算交集和并集
    inter = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)

    # 计算Dice系数
    dice = (2. * inter + epsilon) / (union + epsilon)
    return dice.item()  # 返回为Python标量

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

set_seed(42)

class SynchronizedTransform:
    """同步数据增强类，用于同时对图像和掩码进行变换"""
    def __init__(self,flip_prob=0.2,rotation_range=(-15, 15),keep_size=True,input_type='pil'):
        """
        初始化数据增强参数
        :param flip_prob: 翻转概率阈值
        :param rotation_range: 旋转角度范围，tuple (min_angle, max_angle)
        :param keep_size: 是否保持原始尺寸
        :param input_type: 输入类型，'numpy' 或 'pil'
        """
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.keep_size = keep_size
        self.input_type = input_type

    def __call__(self, image, mask):
        """
        对图像和掩码应用同步的数据增强
        :param image: 输入图像
        :param mask: 输入掩码
        :param input_type: 输入类型，'numpy' 或 'pil'
        :return: 增强后的图像和掩码
        """
        # 转换为PIL图像
        if self.input_type == 'numpy':
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray(mask.astype(np.uint8))
        else:
            image_pil, mask_pil = image, mask

        # 保存原始尺寸
        original_size = image_pil.size

        # 随机水平翻转
        if random.random() > self.flip_prob:
            image_pil = ImageOps.mirror(image_pil)
            mask_pil = ImageOps.mirror(mask_pil)

        # 随机垂直翻转
        if random.random() > self.flip_prob:
            image_pil = ImageOps.flip(image_pil)
            mask_pil = ImageOps.flip(mask_pil)

        # 随机旋转
        angle = random.randint(*self.rotation_range)
        image_pil = image_pil.rotate(angle, expand=False, fillcolor=None)
        mask_pil = mask_pil.rotate(angle, expand=False, fillcolor=0)

        # 确保输出尺寸与原始尺寸相同
        if self.keep_size and image_pil.size != original_size:
            image_pil = image_pil.resize(original_size, Image.BICUBIC)
            mask_pil = mask_pil.resize(original_size, Image.NEAREST)

        # 根据输入类型返回相应格式
        if self.input_type == 'numpy':
            return np.array(image_pil), np.array(mask_pil)
        return image_pil, mask_pil

    @staticmethod
    def set_seed(seed):
        """设置随机种子以确保可重复性"""
        random.seed(seed)
        np.random.seed(seed)

class PTEDataset(IterableDataset):
    def __init__(self, base_dir, phase='train', patient_ids=None,
                 image_size=(3456,5184), num_classes=1,
                 augmentations=None, test_mode=False,
                 shuffle=True, seed=42):
        """
        初始化MRI数据集
        :param base_dir: 数据集根目录（包含train/和val/文件夹）
        :param phase: 数据阶段 ['train' | 'val']
        :param patient_ids: 指定使用的病人ID列表（None表示自动检测）
        :param image_size: 输出图像尺寸
        :param num_classes: 分割类别数
        :param augmentations: 数据增强变换（需同时处理image和mask）
        """
        self.base_dir = base_dir
        self.phase = phase
        self.image_size = image_size
        self.num_classes = num_classes
        self.augmentations = augmentations if augmentations else SynchronizedTransform()
        self.test_mode = test_mode  # 是否为测试模式
        
        # 初始化随机数生成器
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # 自动检测病人ID（如果未指定）
        self.patient_dir = os.path.join(base_dir, phase)
        if patient_ids is None:
            self.patient_ids = sorted([
                d for d in os.listdir(self.patient_dir) 
                if os.path.isdir(os.path.join(self.patient_dir, d))
            ])
        else:
            self.patient_ids = patient_ids
            
        # 生成文件路径列表（不立即加载所有路径）
        self.file_pairs = []
        for pid in self.patient_ids:
            patient_path = os.path.join(self.patient_dir, pid)
            for fname in os.listdir(patient_path):
                if fname.endswith('.png') and '_label' not in fname:
                    img_path = os.path.join(patient_path, fname)
                    mask_path = os.path.join(
                        patient_path, 
                        fname.replace('.png', '_label.png')
                    )
                    if self.test_mode or os.path.exists(mask_path):
                        self.file_pairs.append((img_path, mask_path))
        
        # 基础预处理
        self.base_transform_img = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6151, 0.4225, 0.3555], std=[0.2180, 0.2214, 0.2201]) # RGB图像的均值和标准差
            # transforms.Normalize(mean=[0.4678], std=[0.2182]) # Gray图像的均值和标准差
        ])
        self.base_transform_mask = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
    
    def _calculate_global_stats(self):
        """预计算整个数据集的均值和标准差"""
        pixel_values = []
        for img_path in self.image_paths:
            img = Image.open(img_path)
            img = np.array(img) / 255.0
            pixel_values.extend(img.flatten())
        return np.mean(pixel_values), np.std(pixel_values)
    
    def _process_item(self, img_path, mask_path):
        """处理单个样本"""
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 处理掩码
        if self.test_mode:
            mask = Image.new('L', img.size, 0).convert('L')
        else:
            mask = Image.open(mask_path).convert('L')
        
        # 数据增强
        if self.augmentations and self.phase == 'train':
            img, mask = self.augmentations(img, mask)
        # 保存增强后的图像和掩码进行检查
        # img.save(f"augmented_img.png")
        # mask.save(f"augmented_mask.png")
            
        # 基础变换
        img = self.base_transform_img(img)
        mask = self.base_transform_mask(mask)
        mask = (mask * 255).long()
        mask = torch.where(mask > 0, 1, 0).float()

        
        return img, mask

    def __iter__(self):
        """实现迭代式数据加载"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 分片处理
        if worker_info is None:  # 单worker
            file_pairs = self.file_pairs
            if self.shuffle:
                self.rng.shuffle(file_pairs)
        else:  # 多worker分片
            per_worker = int(np.ceil(len(self.file_pairs) / worker_info.num_workers))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_pairs))
            file_pairs = self.file_pairs[start:end]
            if self.shuffle:
                self.rng.shuffle(file_pairs)
        
        # 流式生成样本
        for img_path, mask_path in file_pairs:
            yield self._process_item(img_path, mask_path)

    @classmethod
    def create_test_dataset(cls, base_dir, **kwargs):
        """创建测试集专用方法"""
        return cls(
            base_dir=base_dir,
            phase='',  # phase参数将被test_mode覆盖
            test_mode=True,
            augmentations=None,  # 测试模式通常不需要增强
            **kwargs
        )


# 创建数据集实例时指定图像和掩码的转换
image_transform = transforms.Compose([])
mask_transform = transforms.Compose([])

# 定义数据路径
root_dir = r"/home/data/zgb/datasets/pterygium_on_the_eyeball"
# patient_ids=['0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010']
# train_dataset = PTEDataset(root_dir, phase='train', patient_ids=patient_ids[:8])
# val_dataset = PTEDataset(root_dir, phase='train', patient_ids=patient_ids[8:])

# 直接创建训练集和验证集实例
train_dataset = PTEDataset(
    root_dir,
    phase='train',
    shuffle=True,
    patient_ids=[f"{i:04d}" for i in range(1, 361)] 
)

val_dataset = PTEDataset(
    root_dir,
    phase='train',
    shuffle=False,
    patient_ids=[f"{i:04d}" for i in range(361, 450)] 
)



# 创建训练集和测试集的数据加载器
batch_size = 2
train_loader = DataLoader(train_dataset, 
                         batch_size=batch_size,
                         shuffle=False,  # IterableDataset需关闭DataLoader的shuffle
                         num_workers=4,
                         pin_memory=True,
                         prefetch_factor=4,
                         worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

# 实例化模型
model = EGEUNet(1,3,bridge=True,gt_ds=True)
# 定义损失函数
criterion = ComboLoss(dice_weight=2.0, ce_weight=1.0, focal_weight=0.0, smooth=1e-6, gamma=2.0, alpha=0.25)
# 定义优化器
learning_rate = 1e-3
optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
# 定义学习率调度器为余弦学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
scaler = torch.amp.GradScaler()
num_epochs = 150  # 设置训练轮次
deep_supervision = False

# 使用GPU计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print("Training on", n_gpu, "GPUs.")
else:
    n_gpu = 0
    print("Training on CPU.")

model.to(device)
model = nn.DataParallel(model)

best_val_loss = float('inf')
patience_counter = 0
patience = 50
train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []

# 训练模型
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    train_dice = 0.0
    count_train = 0
    for inputs, labels in train_loader:
        count_train += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            outputs = model(inputs)[-1]
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_dice += dice_coeff(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        # print('loss:', loss.item())
    epoch_loss = running_loss / count_train
    train_dice_scores.append(train_dice / count_train)
    train_losses.append(epoch_loss)
    # 验证阶段
    model.eval()
    val_running_loss = 0.0
    val_dice = 0.0
    count_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            count_val += 1
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[-1]
            val_dice += dice_coeff(outputs, labels)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
    val_epoch_loss = val_running_loss / count_val
    val_losses.append(val_epoch_loss)
    val_dice_scores.append(val_dice/ count_val)
    # 输出训练和测试集的损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f},Train Dice Score: {train_dice/count_train:.4f}, Val Loss: {val_epoch_loss:.4f},Val Dice Score: {val_dice/count_val:.4f}')

    scheduler.step()

    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'UNet_best.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping...")
        break


# 保存最后训练的模型
torch.save(model.state_dict(), 'UNet_last.pth')

# 绘制LOSS和DICE曲线
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_dice_scores, label='Train Dice Score')
plt.plot(val_dice_scores, label='Val Dice Score')
plt.title('Dice Score Curve')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()

# 保存图像
plt.savefig('loss_dice_curve.png')

# 清空显存
torch.cuda.empty_cache()
