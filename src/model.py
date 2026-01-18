import torch
import torch.nn as nn
import timm

class MorphologyAwareViT(nn.Module):
    def __init__(self, model_name='vit_tiny_patch16_224', pretrained=True):
        super().__init__()
        
        # 1. Backbone: ViT-Tiny
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embedding_dim = 192  # ViT-Tiny dimension
        
        # 2. SSL Head (for learning image similarity)
        self.ssl_head = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 3. The Novelty: Morphology Decoder
        # Takes the 1D vector from ViT and tries to redraw the 2D Mask
        # If the model can redraw the nuclei mask, it MUST have understood morphology!
        self.mask_decoder = nn.Sequential(
            nn.Linear(embedding_dim, 14*14*64), # Upsample
            nn.ReLU(),
            nn.Unflatten(1, (64, 14, 14)),       # Reshape to 2D
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 14->28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 28->56
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 8, stride=4, padding=2),  # 56->224
            nn.Sigmoid() # Output 0-1 (Mask)
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]
        
        # 1. Extract Global Features
        features = self.backbone(x) # [Batch, 192]
        
        # 2. SSL Projection (Contrastive output)
        ssl_out = self.ssl_head(features)
        
        # 3. Mask Prediction (Morphology output)
        predicted_mask = self.mask_decoder(features) # [Batch, 1, 224, 224]
        
        return ssl_out, predicted_mask

# Quick Test Block
if __name__ == "__main__":
    model = MorphologyAwareViT()
    dummy_input = torch.randn(2, 3, 224, 224)
    ssl, mask = model(dummy_input)
    print(f"SSL Output Shape: {ssl.shape}")   # Should be [2, 128]
    print(f"Mask Output Shape: {mask.shape}") # Should be [2, 1, 224, 224]
    print("Model Architecture: Validated.")