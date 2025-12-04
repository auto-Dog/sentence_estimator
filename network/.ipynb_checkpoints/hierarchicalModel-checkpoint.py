import torch
import torch.nn as nn
from transformers import PreTrainedModel
from utils.patch_encoder import bchw_to_pixel_values  # 确保该函数无状态（纯函数最佳）
from transformers import PretrainedConfig

class HierarchicalModelConfig(PretrainedConfig):
    model_type = "hierarchical_vlm"

    def __init__(
        self,
        vlm_config=None,      # 主 VLM 的 config（如 Qwen2VLConfig）
        filter_trainable=True,
        vlm_trainable=False,
        simulate_trainable=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vlm_config = vlm_config
        self.filter_trainable = filter_trainable
        self.simulate_trainable = simulate_trainable
        self.vlm_trainable = vlm_trainable

class HierarchicalModel(PreTrainedModel):
    config_class = HierarchicalModelConfig
    base_model_prefix = "hierarchical_model"
    _no_split_modules = ["ColorFilter", "CVDSimulator"]  # 若有大模块，防止 DDP split

    def __init__(self, vlm_model, color_filter, cvd_simulator, config=None):
        if config is None:
            config = HierarchicalModelConfig(
                        vlm_config=None,
                        filter_trainable=True,
                        vlm_trainable=False,
                        simulate_trainable=False,
                    )
        super().__init__(config)
        self.config = config
        
        # 主模型（如 Qwen2VLForConditionalGeneration）
        self.vlm_model = vlm_model
        
        # 可学习模块（确保是 nn.Module）
        self.color_filter = color_filter
        self.cvd_simulator = cvd_simulator
        
        # 可选：冻结部分模块
        if not config.filter_trainable:
            for p in self.color_filter.parameters():
                p.requires_grad = False
        if not config.simulate_trainable:
            for p in self.cvd_simulator.parameters():
                p.requires_grad = False
        if not config.vlm_trainable:
            for p in self.vlm_model.parameters():
                p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 若需从已有 checkpoint 加载，可自定义；否则用默认即可
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def forward(
        self,
        ori_images=None,       # [B, C, H, W]，原始输入图像 —— 仅用于预处理
        input_ids=None,
        attention_mask=None,
        labels=None,           # ← SFTTrainer 必需！
        pixel_values=None,     # 可选：若已有预处理图像可跳过
        **kwargs,
    ):
        # ✅ 关键：动态处理设备/类型迁移（避免硬编码）
        if ori_images is not None:
            # 自动迁移设备/dtype（兼容 accelerate）
            ori_images = ori_images.to(dtype=self.vlm_model.dtype)
            
            # 预处理流水线
            processed_image = self.color_filter(ori_images)
            processed_image = self.cvd_simulator(processed_image)
            
            # 编码为 VLM 所需 tokens
            pixel_values, image_grid_thw = self._encode_images(processed_image)
            kwargs["pixel_values"] = pixel_values
            if "image_grid_thw" not in kwargs:
                kwargs["image_grid_thw"] = image_grid_thw

        # 转发给主 VLM（确保 labels 透传）
        outputs = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,      # ← 计算 loss 的关键！
            **kwargs
        )
        outputs["ori_images"] = ori_images
        outputs["enhance_images"] = processed_image
        return outputs 

    @torch.no_grad()
    def generate(self, ori_images=None, **kwargs):
        if ori_images is not None:
            ori_images = ori_images.to(dtype=self.vlm_model.dtype)
            processed = self.color_filter(ori_images)
            processed = self.cvd_simulator(processed)
            pixel_values, grid = self._encode_images(processed)
            kwargs["pixel_values"] = pixel_values
            kwargs.setdefault("image_grid_thw", grid)
        return self.vlm_model.generate(**kwargs)

    def _encode_images(self, images):
        # 确保该函数是纯函数（无参数），或将其封装为 nn.Module
        return bchw_to_pixel_values(
            images,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2
        )

    # ✅ 重要：实现标准 HF 接口
    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        # 主模型自动保存；若 color_filter/CVD 是自定义模块，也应能随 state_dict 保存

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the underlying VLM.
        """
        if hasattr(self.vlm_model, "gradient_checkpointing_enable"):
            self.vlm_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        else:
            raise ValueError(
                f"Underlying model {type(self.vlm_model)} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the underlying VLM.
        """
        if hasattr(self.vlm_model, "gradient_checkpointing_disable"):
            self.vlm_model.gradient_checkpointing_disable()
        else:
            # 如果不支持，静默忽略（或 warning）
            pass

    @property
    def supports_gradient_checkpointing(self):
        # 让 Trainer 知道本模型“支持”（实际是底层支持）
        return getattr(self.vlm_model, "supports_gradient_checkpointing", False)

    # 可选：加一个快捷属性（某些代码会检查）
    @property
    def gradient_checkpointing(self):
        return getattr(self.vlm_model, "gradient_checkpointing", False)