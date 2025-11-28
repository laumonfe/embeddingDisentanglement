# To enforce disentanglement architecturally (not just via loss), you can:

# Separate latent spaces:
# Use two distinct bottleneck layers for vision and text, so each modality projects into its own latent space before any interaction.

# No direct concatenation:
# Instead of concatenating, keep the representations separate and only interact via a controlled mechanism (e.g., cross-attention, gating, or a small shared layer).

# Orthogonalization layers:
# Add layers that explicitly decorrelate the two representations (e.g., via Gram-Schmidt or orthogonal projection).

# Modality-specific heads:
# Add separate heads for each modality, and only allow interaction at a late stage (or not at all).

# Example: Separate bottlenecks and orthogonalization


class DisentangledCLIP(nn.Module):
    def __init__(self, vision_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # Bottleneck layers for each modality
        self.vision_bottleneck = nn.Linear(vision_encoder.config.hidden_size, embed_dim)
        self.text_bottleneck = nn.Linear(text_encoder.config.hidden_size, embed_dim)
        # Optional: orthogonalization layer
        self.ortho_layer = nn.Linear(embed_dim, embed_dim, bias=False)
        # Heads for each modality
        self.vision_head = nn.Linear(embed_dim, embed_dim)
        self.text_head = nn.Linear(embed_dim, embed_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        vision_pooled = vision_outputs.pooler_output
        vision_latent = self.vision_bottleneck(vision_pooled)
        # Orthogonalize vision latent
        vision_latent_ortho = self.ortho_layer(vision_latent)

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output
        text_latent = self.text_bottleneck(text_pooled)
        # Orthogonalize text latent
        text_latent_ortho = self.ortho_layer(text_latent)

        # Separate heads
        vision_out = self.vision_head(vision_latent_ortho)
        text_out = self.text_head(text_latent_ortho)

        # No direct concatenation or mixing
        return {
            "vision_out": vision_out,
            "text_out": text_out,
            "vision_latent": vision_latent_ortho,
            "text_latent": text_latent_ortho
        }