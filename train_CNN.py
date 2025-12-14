import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

# 1. Reproducibility
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 2. Dataset
class CaptchaDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["image"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_str = str(row["label"]).zfill(4)
        label = torch.tensor([int(c) for c in label_str], dtype=torch.long)
        return img, label


# 3. Transforms (no resizing)
transform = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ]
)


# 4. Model
class CaptchaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # → (32, 38, 72)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (32, 19, 36)
            nn.Conv2d(32, 64, 3, padding=1),  # → (64, 19, 36)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (64, 9, 18)
            nn.Conv2d(64, 128, 3, padding=1),  # → (128, 9, 18)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # → (128, 4, 9)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),  # → 128*4*9=4608
            nn.Linear(4608, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * 10),  # 4 digits × 10 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)  # → (B, 40)
        return x.view(-1, 4, 10)  # → (B, 4, 10)


def main():
    # 5. Setup
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    dataset = CaptchaDataset("augrument.csv", transform=transform)
    n_train = int(0.8 * len(dataset))
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = CaptchaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    # 6. Training Loop
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # (B, 4, 10)
            loss = sum(criterion(outputs[:, i], labels[:, i]) for i in range(4))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        # 7. Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [ Val ]"):
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=2)  # (B, 4)
                correct += (preds == labels).all(dim=1).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:5.2f}%")

        # 8. Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_captcha_cnn.pth")

    print("Training complete. Best val acc: {:.2f}%".format(best_val_acc * 100))


if __name__ == "__main__":
    main()
