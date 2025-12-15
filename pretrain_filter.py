import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from network.colorFilter import colorFilter
from network.ssim import SSIMLoss
import argparse

# 自定义数据集类，用于加载图像
class ImageDataset(Dataset):
    def __init__(self, json_file, image_root_dir=None, transform=None):
        """
        Args:
            json_file (string): JSON文件路径，包含图像信息
            image_root_dir (string): 图像文件的根目录路径
            transform (callable, optional): 可选的变换操作
        """
        self.image_root_dir = image_root_dir
        self.transform = transform
        
        # 加载JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f] if json_file.endswith('.jsonl') else json.load(f)
            
        # 如果是列表形式，保持原样；如果是字典形式，提取训练数据
        if isinstance(self.data, dict) and 'train' in self.data:
            self.data = self.data['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取样本
        sample = self.data[idx]
        
        # 获取图像路径
        if 'image' in sample:
            image_path = sample['image']
        else:
            raise KeyError(f"Sample at index {idx} does not contain 'image' field")
            
        # 处理图像路径
        if self.image_root_dir:
            image_path = os.path.join(self.image_root_dir, image_path)
            
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个默认图像
            image = Image.new('RGB', (224, 224), color='black')
            
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        # 输入和目标图像是同一张图像（保持不变的目标）
        return image, image

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])
    
    # 创建数据集和数据加载器
    dataset = ImageDataset(
        json_file=args.dataset_path,
        image_root_dir=args.image_root_dir,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = colorFilter().to(device)
    
    # 定义损失函数和优化器
    ssim_loss = SSIMLoss().to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    model.train()
    for epoch in range(args.epochs):
        total_ssim_loss = 0.0
        total_mse_loss = 0.0
        num_batches = 0
        
        for batch_idx, (input_images, target_images) in enumerate(dataloader):
            input_images = input_images.to(device)
            target_images = target_images.to(device)
            
            # 前向传播
            output_images = model(input_images)
            
            # 计算损失
            loss_ssim = ssim_loss(output_images, target_images)
            loss_mse = mse_loss(output_images, target_images)
            loss = loss_ssim + 0.1 * loss_mse  # 结合SSIM和MSE损失
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_ssim_loss += loss_ssim.item()
            total_mse_loss += loss_mse.item()
            num_batches += 1
            
            # 打印批处理信息
            if batch_idx % args.log_interval == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx}/{len(dataloader)}], '
                      f'SSIM Loss: {loss_ssim.item():.4f}, MSE Loss: {loss_mse.item():.4f}')
        
        # 打印每个epoch的平均损失
        avg_ssim_loss = total_ssim_loss / num_batches
        avg_mse_loss = total_mse_loss / num_batches
        print(f'Epoch [{epoch+1}/{args.epochs}] Average - SSIM Loss: {avg_ssim_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}')
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f'filter_checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ssim_loss': avg_ssim_loss,
                'mse_loss': avg_mse_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint: {checkpoint_path}')
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'pretrained_filter.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model: {final_model_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain image filter with SSIM loss")
    parser.add_argument("--dataset_path", type=str, required=True, default="./color_150k.json",
                        help="Path to the JSON dataset file")
    parser.add_argument("--image_root_dir", type=str, default=None,
                        help="Root directory for images (if paths in JSON are relative)")
    parser.add_argument("--output_dir", type=str, default="./pretrained_filters",
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log interval for printing training status")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行主函数
    main(args)
