import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_metrics(y_true, y_pred, class_names):
    """Computes comprehensive evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per class accuracy
    per_class_acc = {}
    if len(cm) > 1: # Avoid division by zero if only one class predicted
        cm_diag = cm.diagonal()
        cm_sum = cm.sum(axis=1)
        for i, class_name in enumerate(class_names):
            if i < len(cm_sum) and cm_sum[i] > 0:
                per_class_acc[class_name] = cm_diag[i] / cm_sum[i]
            else:
                per_class_acc[class_name] = 0.0
    
    cr_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    
    return {
        'accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report_str': cr_str
    }

class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        
        lr = config.get('lr', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_weights = copy.deepcopy(self.model.state_dict())
        self.epochs_no_improve = 0
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
        
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
        
    def evaluate(self, test_loader, class_names=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        if class_names is None:
            num_classes = len(np.unique(all_labels))
            class_names = [f"Class {i}" for i in range(num_classes)]
            
        metrics = compute_metrics(all_labels, all_preds, class_names)
        metrics['predictions'] = all_preds
        metrics['true_labels'] = all_labels
        
        return metrics
        
    def fit(self, train_loader, val_loader, num_epochs, patience, model_save_path=None):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                      
            # Early Stopping and Model Saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
                
                if model_save_path:
                    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
                    save_dict = {
                        'model_state': self.best_model_weights,
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }
                    torch.save(save_dict, model_save_path)
            else:
                self.epochs_no_improve += 1
                
            if patience is not None and self.epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        print(f"Training complete. Best Val Loss: {self.best_val_loss:.4f}. Loading best weights.")
        self.model.load_state_dict(self.best_model_weights)
        return self.history
        
    def fine_tune(self, train_loader, val_loader, num_epochs, lr, patience, freeze_features=False, model_save_path=None):
        print(f"Starting fine-tuning...")
        if freeze_features and hasattr(self.model, 'freeze_features'):
            self.model.freeze_features()
            print("Feature extractor frozen.")
        elif hasattr(self.model, 'unfreeze_features'):
            self.model.unfreeze_features()
            
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        weight_decay = self.config.get('weight_decay', 1e-4)
        self.optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
        
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        return self.fit(train_loader, val_loader, num_epochs, patience, model_save_path)
