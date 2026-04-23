import os
import torch
import librosa
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Resize or pad to fixed size, e.g., 128 x 174
    if mel_spec_db.shape[1] < 174:
        pad_width = 174 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :174]
    return mel_spec_db.astype(np.float32)

# -----------------------------
# CNN Model
# -----------------------------
class CNNNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 5 * 21, 256)  # Adjust size based on pooling (after 3 pools: 40/8 x 174/8)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# Training + Validation Function
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30, device='cpu', patience=5):
    """
    Updated with LR Scheduler and Early Stopping.
    """
    model.to(device)
    best_val_acc = 0.0
    counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training with Early Stopping (patience={patience})...")

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress + LR (scheduler ki wajah se)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}", end="")

        # Early Stopping + Best Model Saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
            print(" → Best model saved!")
        else:
            counter += 1
            print(f" (No improvement, counter={counter}/{patience})")
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

        # SCHEDULER STEP (YAHAN DALNA HAI - har epoch ke end mein!)
        scheduler.step(val_acc)  # Val acc ke basis pe LR reduce karega agar 3 epochs mein improve na ho

    # Plot Training Curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    return model, history

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, test_loader, device='cpu', class_names=None):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100

    print("\n=== Final Evaluation on Test Set ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    if class_names:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Test Set)')
        plt.show()

    return accuracy, precision, recall, f1, cm

# -----------------------------
# Dataset Class
# -----------------------------
import librosa.effects

class UrbanSoundDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.noise = T.Vol(gain=0.2)
        self.pitch_shift = T.PitchShift(sample_rate=22050, n_steps=2)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        y, sr = librosa.load(self.file_paths[idx], sr=22050)

        if self.augment:
            # Convert to tensor with batch dim
            y_tensor = torch.from_numpy(y).float().unsqueeze(0)  # [1, length]

            # Apply torchaudio augmentations
            if np.random.rand() < 0.5:
                y_tensor = self.noise(y_tensor)

            if np.random.rand() < 0.3:
                y_tensor = self.pitch_shift(y_tensor)

            # Detach and convert back to numpy (fix for requires_grad)
            y = y_tensor.detach().squeeze(0).numpy()  # Detach here!

            # Time stretch with librosa (on numpy)
            if np.random.rand() < 0.3:
                stretch_rate = np.random.uniform(0.8, 1.2)
                y = librosa.effects.time_stretch(y, rate=stretch_rate)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=1024, hop_length=512)

        if mfcc.shape[1] < 174:
            pad_width = 174 - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :174]

        return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.long)
# -----------------------------
# Load Data (Train + Val + Test)
# -----------------------------
def load_data(base_dir, metadata_csv):
    df = pd.read_csv(metadata_csv)

    # Train: folds 1-8
    train_df = df[df['fold'].isin([1,2,3,4,5,6,7,8])]
    # Validation: fold 9
    val_df = df[df['fold'] == 9]
    # Test: fold 10
    test_df = df[df['fold'] == 10]

    train_files = [os.path.join(base_dir, f'fold{row.fold}/{row.slice_file_name}') for _, row in train_df.iterrows()]
    train_labels = train_df['classID'].tolist()

    val_files = [os.path.join(base_dir, f'fold{row.fold}/{row.slice_file_name}') for _, row in val_df.iterrows()]
    val_labels = val_df['classID'].tolist()

    test_files = [os.path.join(base_dir, f'fold{row.fold}/{row.slice_file_name}') for _, row in test_df.iterrows()]
    test_labels = test_df['classID'].tolist()

    class_names = sorted(df['class'].unique().tolist())

    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), class_names

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    base_dir = '/content/UrbanSound8K/audio'
    metadata_csv = '/content/UrbanSound8K/metadata/UrbanSound8K.csv'

    # Load data
    # Load data
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), class_names = load_data(base_dir, metadata_csv)
     

    train_dataset = UrbanSoundDataset(train_files, train_labels, augment=True)
    val_dataset   = UrbanSoundDataset(val_files, val_labels, augment=False)
    test_dataset  = UrbanSoundDataset(test_files, test_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    num_classes = len(class_names)
    model = CNNNetwork(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # Weight decay badha diya
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)  # verbose=False fix

    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30, device=device)

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth"))

    print("Evaluating on Test Set (Fold 10)...")
    evaluate_model(model, test_loader, device=device, class_names=class_names)
