import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class VisionEmotionModel(nn.Module):
    def __init__(self, num_classes):
        super(VisionEmotionModel, self).__init__()
        # Load lightweight pretrained MobileNetV3
        self.base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        
        # Replace the final classification layer
        in_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)
