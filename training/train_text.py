import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.text_model import TextEmotionModel
from utils.config import NUM_CLASSES, TEXT_MODEL_NAME, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR

# Dummy dataset for illustration (In a real scenario, load GoEmotions)
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextEmotionModel(NUM_CLASSES).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_NAME)
    
    # Placeholder data
    train_data = TextDataset(["I am happy", "I am sad"], [3, 4], tokenizer, MAX_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Text Model Training...")
    for epoch in range(1): # Just 1 epoch for mock
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")
            
    print("Saving model...")
    torch.save(model.state_dict(), MODEL_DIR / "text_model.pth")
    print("Done!")

if __name__ == "__main__":
    train()
