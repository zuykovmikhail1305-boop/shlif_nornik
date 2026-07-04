import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

# ------------------- НАСТРОЙКИ -------------------
DATA_DIR = 'C://Users/User/Desktop/hak/shlif_nornik/train_cnn'    # <-- замените на свой путь
MODEL_SAVE_PATH = 'grayscale_512_cnn_pytorch.pth'

IMG_SIZE = 512           # УМЕНЬШЕНО до 512x512 (было 1000x1000)
BATCH_SIZE = 8           # Можно увеличить батч (было 2) - 512x512 весит в ~4 раза меньше
EPOCHS = 20
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# ------------------- ТРАНСФОРМАЦИИ -------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# ------------------- ЗАГРУЗКА ДАННЫХ -------------------
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Классы: {class_names}")
print(f"Всего изображений: {len(full_dataset)}")

val_size = int(len(full_dataset) * VALIDATION_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Тренировочных: {train_size}, валидационных: {val_size}")

# ------------------- МОДЕЛЬ CNN (для 512x512) -------------------
class CNN512(nn.Module):
    def __init__(self, num_classes=3):
        super(CNN512, self).__init__()
        # Блок 1: 512 -> 256
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Блок 2: 256 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Блок 3: 128 -> 64
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Блок 4: 64 -> 32
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # Глобальный средний пулинг и классификатор
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = CNN512(num_classes=num_classes).to(DEVICE)

# ------------------- ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАТОР -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------- ОБУЧЕНИЕ -------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

for epoch in range(1, EPOCHS+1):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    print(f'Epoch {epoch:02d}/{EPOCHS}: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# ------------------- СОХРАНЕНИЕ МОДЕЛИ (ПРАВИЛЬНОЕ ДЛЯ ПРОДА) -------------------
checkpoint = {
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
    'num_classes': num_classes,
    'img_size': IMG_SIZE,
    'model_architecture': 'CNN512',
    'device': str(DEVICE),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
}
torch.save(checkpoint, MODEL_SAVE_PATH)
print(f'Чекпоинт сохранён в {MODEL_SAVE_PATH}')
print(f'Классы: {class_names}')

# ------------------- ФУНКЦИЯ ЗАГРУЗКИ МОДЕЛИ -------------------
def load_model_for_inference(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if checkpoint.get('model_architecture') != 'CNN512':
        raise ValueError("Архитектура модели не соответствует ожидаемой!")
    
    num_classes = checkpoint['num_classes']
    loaded_model = CNN512(num_classes=num_classes).to(device)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    metadata = {
        'class_names': checkpoint['class_names'],
        'img_size': checkpoint['img_size'],
        'num_classes': num_classes,
    }
    return loaded_model, metadata

# ------------------- ФУНКЦИЯ ПРЕДСКАЗАНИЯ -------------------
def predict_image(img_path, model, metadata, device):
    from PIL import Image
    
    model.eval()
    img_size = metadata['img_size']
    class_names = metadata['class_names']
    
    image = Image.open(img_path).convert('L')
    image = image.resize((img_size, img_size))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(prob, 1)
    
    predicted_class = class_names[pred_idx.item()]
    confidence = conf.item()
    
    all_probs = {
        class_names[i]: prob[0][i].item()
        for i in range(len(class_names))
    }
    
    return predicted_class, confidence, all_probs

# ------------------- ТЕСТ -------------------
loaded_model, metadata = load_model_for_inference(MODEL_SAVE_PATH, DEVICE)
print(f"Загружена модель для классов: {metadata['class_names']}")
print(f"Ожидаемый размер изображения: {metadata['img_size']}x{metadata['img_size']}")

# cls, conf, all_probs = predict_image('test_image.png', loaded_model, metadata, DEVICE)
# print(f"Класс: {cls}, уверенность: {conf:.4f}")
# print(f"Все вероятности: {all_probs}")

# ------------------- ГРАФИКИ -------------------
epochs_range = range(1, EPOCHS+1)
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(epochs_range, history['train_acc'], label='Train')
plt.plot(epochs_range, history['val_acc'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, history['train_loss'], label='Train')
plt.plot(epochs_range, history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()