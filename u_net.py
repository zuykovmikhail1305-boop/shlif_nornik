import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, jaccard_score

# Разрешаем Pillow загружать «битые» (truncated) изображения
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------- Модель U-Net ----------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.out_conv(x)

# ---------- Датасет ----------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(512, 512), augment=False):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment

        self.img_paths = sorted([p for p in self.img_dir.rglob('*') 
                                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')])
        self.mask_paths = sorted([p for p in self.mask_dir.rglob('*') 
                                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')])

        assert len(self.img_paths) == len(self.mask_paths), \
            f"Разное количество изображений ({len(self.img_paths)}) и масок ({len(self.mask_paths)})"
        print(f"Загружено {len(self.img_paths)} пар изображений и масок.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
        mask = (mask > 0.5).float()

        return image, mask

# ---------- Функция потерь ----------
class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        inputs_sigmoid = torch.sigmoid(inputs)
        smooth = 1e-6
        intersection = (inputs_sigmoid * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (inputs_sigmoid.sum() + targets.sum() + smooth)
        return self.weight_bce * bce + self.weight_dice * dice

# ---------- Метрики ----------
def compute_metrics(outputs, targets, threshold=0.5):
    """
    Вычисляет основные метрики бинарной сегментации.
    Возвращает словарь с метриками для текущего батча (средние по батчу).
    """
    outputs_sigmoid = torch.sigmoid(outputs)
    preds = (outputs_sigmoid > threshold).float()
    
    # Переводим в numpy для sklearn
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    
    # Pixel Accuracy
    acc = accuracy_score(targets_np, preds_np)
    
    # Precision, Recall, F1 (Dice), IoU
    prec = precision_score(targets_np, preds_np, zero_division=0)
    rec = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)
    iou = jaccard_score(targets_np, preds_np, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'dice': f1,
        'iou': iou
    }

# ---------- Обучение ----------
def train(model, train_loader, val_loader, epochs, device, save_path='unet_best.pth'):
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Валидация
        model.eval()
        val_loss = 0
        metrics_sum = {'accuracy': 0, 'precision': 0, 'recall': 0, 'dice': 0, 'iou': 0}
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                batch_metrics = compute_metrics(outputs, masks)
                for k in metrics_sum:
                    metrics_sum[k] += batch_metrics[k]

        avg_val_loss = val_loss / len(val_loader)
        for k in metrics_sum:
            metrics_sum[k] /= len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        history['val_iou'].append(metrics_sum['iou'])
        history['val_dice'].append(metrics_sum['dice'])

        print(f'Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} | '
              f'IoU: {metrics_sum["iou"]:.4f} | '
              f'Dice: {metrics_sum["dice"]:.4f} | '
              f'Precision: {metrics_sum["precision"]:.4f} | '
              f'Recall: {metrics_sum["recall"]:.4f} | '
              f'Accuracy: {metrics_sum["accuracy"]:.4f}')

        # Планировщик lr
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f'Learning rate reduced to {new_lr:.2e}')

        # Сохранение лучшей модели по IoU
        if metrics_sum['iou'] > best_iou:
            best_iou = metrics_sum['iou']
            torch.save(model.state_dict(), save_path)
            print(f'Сохранена лучшая модель (IoU = {best_iou:.4f})')

    # Графики
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history['val_iou'], label='Val IoU')
    plt.plot(epochs_range, history['val_dice'], label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.subplot(2, 2, 3)
    # Если захотим сохранять lr, раскомментируйте:
    # plt.plot(epochs_range, [lr for lr in history.get('lr', [])])
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return history

# ---------- Запуск ----------
if __name__ == '__main__':
    IMG_DIR = '/home/team056/training_sigment/картинки'
    MASK_DIR = '/home/team056/training_sigment/маски'

    BATCH_SIZE = 4
    EPOCHS = 30
    IMG_SIZE = (512, 512)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VAL_SPLIT = 0.2  # 20% на валидацию

    print(f'Используется устройство: {DEVICE}')

    full_dataset = SegmentationDataset(IMG_DIR, MASK_DIR, img_size=IMG_SIZE, augment=True)
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1)
    print(f'Количество параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    train(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE, save_path='unet_best.pth')