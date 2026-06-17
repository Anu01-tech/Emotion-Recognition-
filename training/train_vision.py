import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.vision_model import VisionEmotionModel
from utils.config import NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR

class VisionDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # Mock data: (channels=3, width=224, height=224)
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, NUM_CLASSES, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, item):
        return self.data[item], self.labels[item]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VisionEmotionModel(NUM_CLASSES).to(device)
    
    train_data = VisionDataset()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Vision Model Training...")
    for epoch in range(1):
        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print("Saving model...")
    torch.save(model.state_dict(), MODEL_DIR / "vision_model.pth")
    print("Done!")

if __name__ == "__main__":
    train()
