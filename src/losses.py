import torch 
import torch.nn as nn

def contrastive_loss(image_embeds, text_embeds, group_indices=None,  temperature=0.07):
    # Normalize
    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.t(), labels)
    return (loss_img + loss_txt) / 2


def grouped_contrastive_loss(image_embeds, text_embeds, group_indices, temperature=0.07):
    """
    image_embeds: [B, D] (B = batch size, one image per batch)
    text_embeds: [N, D] (N = total number of captions in batch)
    group_indices: list of lists, each sublist contains indices of captions for each image
    """
    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature  # [B, N]

    # Build targets: for each image, all its captions are positives
    targets = torch.zeros_like(logits)
    for i, indices in enumerate(group_indices):
        targets[i, indices] = 1  # positives

    # Use BCEWithLogitsLoss for multi-label
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    return loss


def disentangled_clip_loss(image_embeds, text_embeds, group_indices=None, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1):
    D = text_embeds.shape[1]
    D_c = D // 2  # first half: content, second half: subjective
    text_content = text_embeds[:, :D_c]
    text_subjective = text_embeds[:, D_c:]

    image_embeds_norm = nn.functional.normalize(image_embeds, dim=-1)
    text_content_norm = nn.functional.normalize(text_content, dim=-1)
    image_content = image_embeds_norm[:, :D_c]
    logits = image_content @ text_content_norm.t() / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_content_img = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2

    text_subjective_norm = nn.functional.normalize(text_subjective, dim=-1)
    image_subjective = image_embeds_norm[:, D_c:]
    loss_subjective_img = (image_subjective * text_subjective_norm).sum(dim=1).abs().mean()
    loss_subjective_content = (text_content_norm * text_subjective_norm).sum(dim=1).abs().mean()

    loss = alpha * loss_content_img + beta * loss_subjective_img + gamma * loss_subjective_content
    return loss