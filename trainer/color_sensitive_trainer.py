# color_sensitive_trainer.py
import torch
from swift import Seq2SeqTrainer  # ms-swift 训练器

class ColorSensitiveTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        importance_mask = inputs.pop("importance_mask", None)
        outputs = model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if importance_mask is not None:
            shift_importance = importance_mask[..., 1:].contiguous()
        else:
            shift_importance = torch.ones_like(shift_labels, dtype=torch.float)

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        weighted_loss = (per_token_loss * shift_importance.view(-1)).sum() / shift_importance.sum()

        return (weighted_loss, outputs) if return_outputs else weighted_loss
