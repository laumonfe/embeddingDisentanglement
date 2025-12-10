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
    
    # Deterministic, non-trainable projection
    if image_embeds_norm.shape[1] != D_c:
        proj_matrix = get_fixed_projection(image_embeds_norm.shape[1], D_c, image_embeds_norm.device)
        image_content = image_embeds_norm @ proj_matrix  # [B, D_c]
    else:
        image_content = image_embeds_norm

    logits = image_content @ text_content_norm.t() / temperature  # [B, N]

    # Multi-label targets: for each image, all its captions are positives
    targets = torch.zeros_like(logits)
    for i, indices in enumerate(group_indices):
        targets[i, indices] = 1

    loss_content_img = nn.BCEWithLogitsLoss()(logits, targets)

    text_subjective_norm = nn.functional.normalize(text_subjective, dim=-1)
    # Subjective losses only use text_subjective
    loss_subjective_content = (text_content_norm * text_subjective_norm).sum(dim=1).abs().mean()

    loss = alpha * loss_content_img + gamma * loss_subjective_content
    return loss


def grouped_disentangled_loss(image_embeds, text_embeds, group_indices, temperature=0.07, alpha=1.0, beta=1.0, gamma=0.1):
    D = text_embeds.shape[1]
    D_c = D // 2
    text_content = text_embeds[:, :D_c]
    text_subjective = text_embeds[:, D_c:]

    image_embeds_norm = nn.functional.normalize(image_embeds, dim=-1)
    text_content_norm = nn.functional.normalize(text_content, dim=-1)

    # Deterministic, non-trainable projection
    if image_embeds_norm.shape[1] != D_c:
        proj_matrix = get_fixed_projection(image_embeds_norm.shape[1], D_c, image_embeds_norm.device)
        image_content = image_embeds_norm @ proj_matrix  # [B, D_c]
    else:
        image_content = image_embeds_norm

    logits = image_content @ text_content_norm.t() / temperature

    targets = torch.zeros_like(logits)
    for i, indices in enumerate(group_indices):
        targets[i, indices] = 1

    loss_content_img = nn.BCEWithLogitsLoss()(logits, targets)
    text_subjective_norm = nn.functional.normalize(text_subjective, dim=-1)
    loss_subjective_content = (text_content_norm * text_subjective_norm).sum(dim=1).abs().mean()
    loss = alpha * loss_content_img + gamma * loss_subjective_content
    return loss


def get_fixed_projection(in_dim, out_dim, device):
    torch.manual_seed(42)  # for reproducibility
    proj_matrix = torch.randn(in_dim, out_dim, device=device)
    proj_matrix = nn.functional.normalize(proj_matrix, dim=0)
    return proj_matrix