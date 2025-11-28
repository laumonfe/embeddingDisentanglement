from transformers import Trainer


# Define a custom Trainer to access both vision and text models
class CustomCLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        # CLIPModel returns logits_per_image and logits_per_text
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss