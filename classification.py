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
def extract_features(file_path, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=22050)
    n_fft = 1024
    hop_length = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, 
                               n_fft=n_fft, hop_length=hop_length)
    
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.astype(np.float32)

# -----------------------------
# CNN Model
# -----------------------------
class CNNNetwork(nn.Module):
    def __init__(self, num_classes):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 10 * 43, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -----------------------------
# Training + Validation Function
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30, device='cpu'):
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
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

        # Validation
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

        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  → Best model saved! Val Acc: {val_acc:.2f}%")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

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
class UrbanSoundDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)  # Yeh line zaroori hai!

    def __getitem__(self, idx):
        mfcc = extract_features(self.file_paths[idx])
        mfcc = torch.tensor(mfcc).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mfcc, label

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
     
     # Print lengths to debug
    print("Train samples:", len(train_files))
    print("Val samples:", len(val_files))
    print("Test samples:", len(test_files))
    train_dataset = UrbanSoundDataset(train_files, train_labels)
    val_dataset = UrbanSoundDataset(val_files, val_labels)
    test_dataset = UrbanSoundDataset(test_files, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    num_classes = len(class_names)
    model = CNNNetwork(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 1e-4 ya 1e-3 try kar

    print("Starting training...")
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30, device=device)

    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth"))

    print("Evaluating on Test Set (Fold 10)...")
    evaluate_model(model, test_loader, device=device, class_names=class_names)

    # Optional: Test on a single file
    print("Classifying a new audio file...")
    new_audio = '/content/UrbanSound8K/audio/fold10/2937-1-0-0.wav'
    if os.path.exists(new_audio):
        features = extract_features(new_audio)
        features_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(features_tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            predicted_class = class_names[pred_idx]
        print(f"Predicted class: {predicted_class} (index {pred_idx})")
    else:
        print(f"File not found: {new_audio}")
