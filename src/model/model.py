import torch
import torch.nn as nn
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.basicConfig(level=logging.INFO)


class VoiceGrader(nn.Module):
    def __init__(self, input_size, num_filters=256, kernel_size=5, stride=2, dropout=0.5, device="cuda"):
        super(VoiceGrader, self).__init__()

        self.to(device)
        self.device = device
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, stride)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, stride)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * 2, 128)  # Adjust the size based on your needs
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("forwarding...")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=-1)  # Global Average Pooling (GAP) layer
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.clamp(x, min=0, max=100)
        return x

class Trainer:
    def __init__(self, model, optimizer, criterion, num_epochs=10, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True, factor=0.5)

    def train_model(self, train_loader, val_loader, save_path="best_model.pth"):
        best_loss = float('inf')
        self.model.to(self.device)  # Ensure the model is on the correct device

        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            # Training Loop
            self.model.train()
            total_loss = 0
            for spectrograms, scores in train_loader:
                spectrograms, scores = spectrograms.to(self.device), scores.to(self.device)
                outputs = self.model(spectrograms.float())
                loss = self.criterion(outputs.squeeze().float(), scores.float())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # Validation Loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for spectrograms, scores in val_loader:
                    spectrograms, scores = spectrograms.to(self.device), scores.to(self.device)
                    outputs = self.model(spectrograms.float())
                    loss = self.criterion(outputs.squeeze().float(), scores.float())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # Update scheduler (based on validation loss)
            self.scheduler.step(avg_val_loss)

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            # Checkpointing
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.model.state_dict(), save_path)

            logging.info(
                f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

