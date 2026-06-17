import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.speech_model import SpeechEmotionModel
from utils.config import NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR

class SpeechDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # Mock data: (channels=1, n_mfcc=40, time_steps=130)
        self.data = torch.randn(size, 1, 40, 130)
        self.labels = torch.randint(0, NUM_CLASSES, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, item):
        return self.data[item], self.labels[item]

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpeechEmotionModel(NUM_CLASSES).to(device)
    
    train_data = SpeechDataset()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Speech Model Training...")
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
    torch.save(model.state_dict(), MODEL_DIR / "speech_model.pth")
    print("Done!")

if __name__ == "__main__":
    train()
