# TODO: Fix BUG
# color_sensitive_trainer.py
import torch
from trl import SFTTrainer
from network import SSIMLoss

class ColorSensitiveTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_loss = SSIMLoss()
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        ori_image_pixels = inputs.get("ori_pixel_values")
        image_pixels = inputs.get("pixel_values")
        ssim_loss = self.ssim_loss(ori_image_pixels, image_pixels)
        loss += ssim_loss
        self._metrics["train"]["ssim_loss"] = ssim_loss.item()
        return (loss, outputs) if return_outputs else loss  
