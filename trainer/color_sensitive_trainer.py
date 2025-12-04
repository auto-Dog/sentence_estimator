# color_sensitive_trainer.py
import torch
from trl import SFTTrainer
from network import SSIMLoss

class ColorSensitiveTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_loss = SSIMLoss()
        self._ssim_step_counter = 0
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        ori_image = outputs.get("ori_images")
        enhance_image = outputs.get("enhance_images")
        # image_pixels = inputs.get("pixel_values") # pixel_values is the reshaped enhance image
        ssim_loss = self.ssim_loss(ori_image, enhance_image)
        ssim_weight = 1 - min(0.8, self._ssim_step_counter / 500)
        self._ssim_step_counter += 1
        loss = (1-ssim_weight) * loss + ssim_weight * ssim_loss
        self._metrics[mode]["ssim_loss"].append(ssim_loss.item())
        return (loss, outputs) if return_outputs else loss  
