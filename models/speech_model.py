import torch
import torch.nn as nn

class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechEmotionModel, self).__init__()
        # Input shape expected: (batch, channels=1, n_mfcc=40, time_steps)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # After 3 MaxPool layers (divide by 8)
        # n_mfcc = 40 -> 40 / 8 = 5
        self.lstm_input_size = 128 * 5
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        
        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 3, 1, 2) # (batch, time, channels, height)
        x = x.reshape(batch_size, x.size(1), -1) # (batch, time, 128*5)
        
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        out = out[:, -1, :]
        
        return self.fc(out)
