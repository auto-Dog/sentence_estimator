import torch
import torch.nn as nn
import kornia.color as kcolor
from colour.blindness import matrix_cvd_Machado2009

class CVDSimulateNetMachado(nn.Module):
    def __init__(self, cvd_type='Deuteranomaly', severity=1.0):
        """可微分的色盲模拟网络
        
        Args:
            cvd_type (str, optional): 色盲类型. Defaults to 'Deuteranomaly'. 
               可选: 'Protanomaly', 'Deuteranomaly', 'Tritanomaly'
            severity (float, optional): 色盲严重程度，0.0-1.0. Defaults to 1.0.
        """
        super().__init__()
        self.cvd_type = cvd_type
        self.severity = severity
        
        # 获取色盲变换矩阵并转换为PyTorch张量
        mat = matrix_cvd_Machado2009(cvd_type, severity)
        self.register_buffer('transform_matrix', torch.from_numpy(mat).float())
    
    def forward(self, **input):
        """
        input: 图像张量，形状可以是:
               - (B, 3, H, W): 批处理的图像，值范围0-1
               - (3, H, W): 单张图像，值范围0-1
               - (H, W, 3): 单张图像，值范围0-1
        output: 处理后的图像张量，与输入相同形状
        """
        original_shape = x.shape
        x = input['pixel_values']
        
        # 确保输入为4D张量 (B, C, H, W)
        if x.dim() == 3:
            if original_shape[-1] == 3:  # (H, W, 3) -> (1, 3, H, W)
                x = x.permute(2, 0, 1).unsqueeze(0)
            else:  # (C, H, W) -> (1, C, H, W)
                x = x.unsqueeze(0)
        elif x.dim() == 4 and original_shape[-1] == 3:  # (B, H, W, 3) -> (B, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        
        # 确保值范围在0-1之间
        if x.max() > 1.0:
            x = x / 255.0
        
        # 1. sRGB转线性RGB [citation:1][citation:5][citation:7]
        linear_rgb = kcolor.rgb_to_linear_rgb(x)
        
        # 2. 应用色盲变换矩阵
        # 重塑为 (B, 3, H*W) 便于矩阵乘法
        B, C, H, W = linear_rgb.shape
        linear_rgb_flat = linear_rgb.view(B, C, -1)
        
        # 应用变换: output = input @ M^T
        # transform_matrix形状: (3, 3)
        transformed_flat = torch.bmm(
            self.transform_matrix.unsqueeze(0).expand(B, -1, -1), 
            linear_rgb_flat
        )
        
        transformed = transformed_flat.view(B, C, H, W)
        
        # 3. 线性RGB转sRGB [citation:1][citation:5][citation:7]
        srgb_transformed = kcolor.linear_rgb_to_rgb(transformed)
        
        # 裁剪到有效范围
        srgb_transformed = torch.clamp(srgb_transformed, 0.0, 1.0)
        
        # 恢复原始形状
        if len(original_shape) == 3:
            if original_shape[-1] == 3:  # 原为 (H, W, 3)
                srgb_transformed = srgb_transformed.squeeze(0).permute(1, 2, 0)
            else:  # 原为 (C, H, W)
                srgb_transformed = srgb_transformed.squeeze(0)
        elif len(original_shape) == 4 and original_shape[-1] == 3:  # 原为 (B, H, W, 3)
            srgb_transformed = srgb_transformed.permute(0, 2, 3, 1)
        input['pixel_values'] = srgb_transformed
        return input
    
    def simulate_pil(self, pil_image):
        """处理PIL图像的便利方法（不可微）
        
        Args:
            pil_image: PIL Image对象
            
        Returns:
            PIL Image: 处理后的图像
        """
        import torchvision.transforms as T
        from PIL import Image
        
        # PIL转Tensor
        transform = T.Compose([
            T.ToTensor()
        ])
        
        with torch.no_grad():
            tensor_img = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
            output_tensor = self.forward(tensor_img)
            
            # Tensor转PIL
            output_tensor = output_tensor.squeeze(0).clamp(0, 1)
            to_pil = T.ToPILImage()
            return to_pil(output_tensor)

# 创建实例
my_observer = CVDSimulateNetMachado(cvd_type='Deuteranomaly', severity=0.5)

# 使用示例
if __name__ == "__main__":
    # 示例1: 处理张量
    dummy_input = torch.rand(1, 3, 224, 224)  # 批处理图像
    output = my_observer(dummy_input)
    print(f"输入形状: {dummy_input.shape}, 输出形状: {output.shape}")
    