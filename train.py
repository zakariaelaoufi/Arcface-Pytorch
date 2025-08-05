import torch
from torch.optim.adam import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.FaceNet import FaceNet
from tqdm import tqdm
from data.DataLoader import train_dataloader, val_dataloader
from PIL import Image
from tqdm import tqdm
import datetime as dt
from sklearn.metrics import accuracy_score


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 540

model = FaceNet(num_classes=num_classes, embedding_dim=512).to(device)
print(f"Model initialized on {device}")


EPOCHS = 4
learning_rate = 1e-3
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

history = {
    'train_loss': [], 'dev_loss': [],
    'train_acc': [], 'dev_acc': [],
    'lr': []
}

resume = True

checkpoint = torch.load('/content/arcface_model_artifact_58', map_location=device)
if resume:
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch']
    else:
        print("No checkpoint found. Starting from scratch.")


start_time = dt.datetime.now()

for epoch in range(EPOCHS):
    # === Training phase ===
    model.train()
    epoch_train_loss = 0
    train_correct = 0
    train_total = 0

    train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    # === Validation phase ===
    model.eval()
    epoch_dev_loss = 0
    dev_correct = 0
    dev_total = 0

    with torch.no_grad():
        dev_loop = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for images, labels in dev_loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            epoch_dev_loss += loss.item()

            preds = torch.argmax(outputs, 1)
            dev_correct += (preds == labels).sum().item()
            dev_total += labels.size(0)

            dev_loop.set_postfix(val_loss=loss.item())

    # === Calculate metrics ===
    train_loss = epoch_train_loss / len(train_dataloader)
    dev_loss = epoch_dev_loss / len(val_dataloader)
    train_acc = train_correct / train_total
    dev_acc = dev_correct / dev_total
    current_lr = optimizer.param_groups[0]['lr']

    # === Update history ===
    history['train_loss'].append(train_loss)
    history['dev_loss'].append(dev_loss)
    history['train_acc'].append(train_acc)
    history['dev_acc'].append(dev_acc)
    history['lr'].append(current_lr)

    # Update scheduler
    scheduler.step(dev_acc)

    # === Print epoch summary ===
    print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {dev_loss:.4f} | Acc: {dev_acc:.4f}")
    print(f"Current Learning Rate: {current_lr:.6f}")
    print("-" * 60)

    # === Save checkpoint every 10 epochs ===
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, f'arcface_model_artifact_{epoch}')

# Training completion
end_time = dt.datetime.now()
print(f"Training completed in: {end_time - start_time}")
torch.save(model.state_dict(), f'arcface_model_final_{EPOCHS + len(history)}.pth')