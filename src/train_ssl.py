import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our custom modules
from dataset import CancerSSLDataset
from model import MorphologyAwareViT

# --- Config ---
BATCH_SIZE = 32      # RTX 4050 should handle 32. If error, lower to 16.
LEARNING_RATE = 3e-4 # Standard for Adam
EPOCHS = 10          # For demo purposes. Real research uses 100+.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "models/ssl_checkpoint.pth"

# --- Helper: SimCLR Contrastive Loss ---
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR Loss).
    It forces the model to bring positive pairs (View1, View2) closer 
    and push different images apart.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        # Concatenate representation
        z = torch.cat([z_i, z_j], dim=0) # [2N, D]
        
        # Calculate similarity matrix
        # sim[i, j] = cosine(z[i], z[j]) / temp
        z_norm = torch.nn.functional.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        
        # Create labels: The partner of i is i + batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(z_i.device)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        sim_matrix.masked_fill_(mask, -9e15) # Fill diagonal with -inf
        
        return self.criterion(sim_matrix, labels)

# --- Main Training Loop ---
def train():
    print(f"Starting Training on: {DEVICE}")
    os.makedirs("models", exist_ok=True)
    
    # 1. Prepare Data
    # Strong augmentations for SSL
    ssl_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor()
    ])
    
    dataset = CancerSSLDataset(root_dir=os.path.join("data", "raw"), transform=ssl_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 2. Prepare Model
    model = MorphologyAwareViT().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Define Losses
    loss_contrastive = NTXentLoss().to(DEVICE)
    loss_reconstruction = nn.MSELoss().to(DEVICE) # Mean Squared Error for Mask
    
    loss_history = []

    # 4. Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (view1, view2, mask) in enumerate(loop):
            # Move data to GPU
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)
            mask = mask.to(DEVICE)
            
            # Forward Pass
            # We get SSL features AND the Predicted Mask
            ssl1, pred_mask1 = model(view1)
            ssl2, pred_mask2 = model(view2)
            
            # Calculate Losses
            # A. Contrastive Loss (Force views to be similar)
            l_ssl = loss_contrastive(ssl1, ssl2)
            
            # B. Morphology Loss (Force predicted mask to match generated mask)
            # We average the loss from both views
            l_morph = (loss_reconstruction(pred_mask1, mask) + loss_reconstruction(pred_mask2, mask)) / 2
            
            # C. Total Loss (Weighted sum)
            # We give equal weight (1.0) to both, or tweak if needed
            loss = l_ssl + (5.0 * l_morph) 
            # Note: Multiplied morph loss by 5 because MSE is usually very small (0.00x) 
            # compared to Contrastive Loss (3.xx), so we balance them.
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), ssl=l_ssl.item(), morph=l_morph.item())
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), SAVE_PATH)

    print("Training Complete! Model saved to 'models/ssl_checkpoint.pth'")
    
    # Plot Loss Curve
    plt.plot(loss_history)
    plt.title("Self-Supervised Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("models/training_curve.png")
    print("Loss curve saved.")

if __name__ == "__main__":
    train()