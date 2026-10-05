import torch
import torch.nn as nn
import os

class CSIActivityCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CSIActivityCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def get_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


def build_model(num_classes=4, device='cpu'):
    model = CSIActivityCNN(num_classes=num_classes)
    model = model.to(device)
    total, trainable = model.get_num_params()
    print(f"Built CSIActivityCNN (Classes: {num_classes}) on {device}")
    print(f"Total parameters: {total:,} | Trainable parameters: {trainable:,}")
    return model


def load_model(path, num_classes=4, device='cpu'):
    model = build_model(num_classes=num_classes, device=device)
    checkpoint = torch.load(path, map_location=device)
    
    # Handle cases where the whole state dict or a dict with 'model_state' is saved
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
        
    print(f"Model loaded from {path}")
    return model


def save_model(model, path, metadata=None):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    save_dict = {'model_state': model.state_dict()}
    if metadata is not None:
        save_dict.update(metadata)
        
    torch.save(save_dict, path)
    print(f"Model saved to {path}")
