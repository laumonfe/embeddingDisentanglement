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
    Multi-positive contrastive loss for grouped data.
    For each image, all its captions are positives; all others are negatives.
    """
    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature  # [B, N]

    # For each image, mask out the positives (all captions in group_indices[i])
    losses = []
    for i, pos_indices in enumerate(group_indices):
        # Numerator: sum over all positives (exp(similarity))
        pos_logits = logits[i, pos_indices]
        numerator = torch.exp(pos_logits).sum()
        # Denominator: sum over all captions
        denominator = torch.exp(logits[i, :]).sum()
        # InfoNCE loss for this image (multi-positive)
        loss_i = -torch.log(numerator / denominator)
        losses.append(loss_i)
    loss = torch.stack(losses).mean()
    return loss


def disentangled_loss(image_embeds, text_embeds, group_indices, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1):
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




def grouped_disentangled_loss(
    image_embeds, text_embeds, group_indices, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1
):
    """
    Grouped disentangled loss:
    - Multi-positive contrastive loss for content alignment (using group_indices)
    - Orthogonality penalties for disentanglement, expanded to match groupings
    """
    D = text_embeds.shape[1]
    D_c = D // 2  # first half: content, second half: subjective
    text_content = text_embeds[:, :D_c]
    text_subjective = text_embeds[:, D_c:]

    image_embeds_norm = nn.functional.normalize(image_embeds, dim=-1)
    text_content_norm = nn.functional.normalize(text_content, dim=-1)
    image_content = image_embeds_norm[:, :D_c]

    # Multi-positive contrastive loss (InfoNCE with multiple positives)
    logits = image_content @ text_content_norm.t() / temperature  # [B, N]
    losses = []
    for i, pos_indices in enumerate(group_indices):
        pos_logits = logits[i, pos_indices]
        numerator = torch.exp(pos_logits).sum()
        denominator = torch.exp(logits[i, :]).sum()
        loss_i = -torch.log(numerator / denominator)
        losses.append(loss_i)
    loss_content_img = torch.stack(losses).mean()

    # Subjective disentanglement: expand image_subjective to match each caption
    text_subjective_norm = nn.functional.normalize(text_subjective, dim=-1)
    image_subjective = image_embeds_norm[:, D_c:]
    expanded_image_subjective = []
    for i, indices in enumerate(group_indices):
        expanded_image_subjective.extend([image_subjective[i]] * len(indices))
    expanded_image_subjective = torch.stack(expanded_image_subjective, dim=0)  # shape [N, D_s]

    # Subjective penalty (orthogonality)
    loss_subjective_img = (expanded_image_subjective * text_subjective_norm).sum(dim=1).abs().mean()
    loss_subjective_content = (text_content_norm * text_subjective_norm).sum(dim=1).abs().mean()

    loss = alpha * loss_content_img + beta * loss_subjective_img + gamma * loss_subjective_content
    return loss
