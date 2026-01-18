import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import our model definition
from model import MorphologyAwareViT

# --- Config ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Lower LR for fine-tuning
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SSL_CHECKPOINT = "models/ssl_checkpoint.pth"

# --- Wrapper Model for Classification ---
class CancerClassifier(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # 1. Initialize the same model architecture
        self.base_model = MorphologyAwareViT()
        
        # 2. Load the Pre-trained Weights (Transfer Learning)
        if pretrained_path:
            print(f"Loading SSL weights from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location=DEVICE)
            # We only want to load the 'backbone' weights, not the old heads
            # We filter out keys that don't match or belong to heads
            model_dict = self.base_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and 'backbone' in k}
            model_dict.update(pretrained_dict)
            self.base_model.load_state_dict(model_dict)
            print("Weights loaded successfully!")
        
        # 3. Define the Classifier Head
        # The backbone outputs 192 dim features (for ViT-Tiny)
        self.classifier = nn.Linear(192, 2) # 2 classes: Benign vs Malignant

    def forward(self, x):
        # We need to bypass the old forward pass which returned (ssl, mask)
        # We just want the backbone features
        features = self.base_model.backbone(x)
        output = self.classifier(features)
        return output

# --- Main Training Loop ---
def train_and_evaluate():
    print(f"Starting Fine-Tuning on: {DEVICE}")
    
    # 1. Data Preparation
    # Standard normalization for ViT
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    # Automatically find train/test folders
    data_dir = os.path.join("data", "raw") 
    # Note: Depending on extraction, it might be 'data/raw/BreaKHis 400X/'
    # Let's search for the folder containing 'train'
    start_dir = None
    for root, dirs, files in os.walk(data_dir):
        if 'train' in dirs and 'test' in dirs:
            start_dir = root
            break
            
    if not start_dir:
        print("Error: Could not locate 'train' and 'test' folders.")
        return

    print(f"Found dataset at: {start_dir}")
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(start_dir, x), data_transforms[x])
                      for x in ['train', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x=='train'), num_workers=2)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")

    # 2. Setup Model
    model = CancerClassifier(pretrained_path=SSL_CHECKPOINT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_acc = 0.0
    train_acc_history = []
    val_acc_history = []

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_acc_history.append(epoch_acc.item())
            else:
                val_acc_history.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy Best Model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'models/best_classifier.pth')

    print(f'Best Test Acc: {best_acc:4f}')
    
    # 4. Final Evaluation & Confusion Matrix
    model.load_state_dict(torch.load('models/best_classifier.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nGenerating Final Report...")
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Graphs and Model saved to 'models/' folder.")

if __name__ == "__main__":
    train_and_evaluate()