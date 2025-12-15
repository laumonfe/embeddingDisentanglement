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




def grouped_disentangled_loss(image_embeds, text_embeds, group_indices, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1):
    D = text_embeds.shape[1]
    D_c = D // 2  # first half: content, second half: subjective
    text_content = text_embeds[:, :D_c]
    text_subjective = text_embeds[:, D_c:]

    image_embeds_norm = nn.functional.normalize(image_embeds, dim=-1)
    text_content_norm = nn.functional.normalize(text_content, dim=-1)
    image_content = image_embeds_norm[:, :D_c]

    # Multi-label contrastive loss (grouped)
    logits = image_content @ text_content_norm.t() / temperature  # [B, N]
    targets = torch.zeros_like(logits)
    for i, indices in enumerate(group_indices):
        targets[i, indices] = 1
    loss_content_img = nn.BCEWithLogitsLoss()(logits, targets)

    text_subjective_norm = nn.functional.normalize(text_subjective, dim=-1)
    image_subjective = image_embeds_norm[:, D_c:]
    # Example: expand image_subjective to match text_subjective_norm
    expanded_image_subjective = []
    for i, indices in enumerate(group_indices):
        # Repeat the i-th image embedding for each associated caption
        expanded_image_subjective.extend([image_subjective[i]] * len(indices))
    expanded_image_subjective = torch.stack(expanded_image_subjective, dim=0)  # shape [N, D_s]

    # Now you can safely compute the penalty
    loss_subjective_img = (expanded_image_subjective * text_subjective_norm).sum(dim=1).abs().mean()  
    loss_subjective_content = (text_content_norm * text_subjective_norm).sum(dim=1).abs().mean()

    loss = alpha * loss_content_img + beta * loss_subjective_img + gamma * loss_subjective_content
    return loss





def covariance_disentanglement_loss(content, subjective):
    """
    Penalizes covariance between content and subjective features to encourage disentanglement.
    """
    # Center the features
    content_centered = content - content.mean(dim=0, keepdim=True)
    subjective_centered = subjective - subjective.mean(dim=0, keepdim=True)
    # Compute covariance matrix between content and subjective
    cov = torch.matmul(content_centered.T, subjective_centered) / (content.shape[0] - 1)
    # Penalize the squared Frobenius norm (sum of squares of all elements)
    return (cov ** 2).mean()