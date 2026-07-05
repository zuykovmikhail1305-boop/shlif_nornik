import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, Response

# Импортируем архитектуру CNN из вашего файла
from shlif_nornik.model_cnn import CNN512

# ---------- НАСТРОЙКИ ----------
MODEL_PATH = 'cnn.pth'                # путь к чекпоинту модели
BLOCK_SIZE = 512                      # размер блока (совпадает с обучением)
USE_DENOISE = True                    # bilateral filter
USE_LIGHT = True                      # CLAHE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {DEVICE}")

# ---------- Загрузка модели ----------
def load_cnn_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get('model_architecture') != 'CNN512':
        raise ValueError("Модель не является CNN512")
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    img_size = checkpoint['img_size']
    model = CNN512(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names, img_size

model, class_names, img_size = load_cnn_model(MODEL_PATH, DEVICE)
print(f"Загружены классы: {class_names}")

# ---------- Функции предобработки (из data_cnn.py) ----------
def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def preprocess_image(img_bgr):
    """Переводит в grayscale, применяет шумоподавление и CLAHE."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if USE_DENOISE:
        gray = apply_bilateral_filter(gray)
    if USE_LIGHT:
        gray = apply_clahe(gray)
    return gray

# ---------- Классификация блока ----------
transform = transforms.Compose([transforms.ToTensor()])

def classify_block(block_gray):
    """Принимает блок grayscale (numpy, uint8), возвращает индекс класса."""
    if block_gray.shape != (img_size, img_size):
        block_gray = cv2.resize(block_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = transform(block_gray).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

# ---------- FastAPI приложение ----------
app = FastAPI(title="Классификация блоков (tonkie/ryadovye)")

HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>Классификация тонных и рядовых блоков</title></head>
<body>
    <h2>Загрузите изображение</h2>
    <form action="/predict/" method="post" enctype="multipart/form-data">
        <p>
            <label>Имя класса для жёлтой обводки (tonkie):</label>
            <input type="text" name="class_tonkie" value="tonkie">
        </p>
        <p>
            <label>Имя класса для зелёной обводки (ryadovye):</label>
            <input type="text" name="class_ryadovye" value="ryadovye">
        </p>
        <p>
            <label>Толщина обводки (пикселей):</label>
            <input type="number" name="thickness" value="3" min="1" max="10">
        </p>
        <p>
            <input type="file" name="file" accept="image/*" required>
        </p>
        <button type="submit">Отправить</button>
    </form>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_FORM

@app.post("/predict/", response_class=Response)
async def predict(
    file: UploadFile = File(...),
    class_tonkie: str = Form("tonkie"),
    class_ryadovye: str = Form("ryadovye"),
    thickness: int = Form(3)
):
    # Чтение изображения
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return Response("Ошибка чтения изображения", status_code=400)

    h, w = img_bgr.shape[:2]

    # Определяем индексы классов
    try:
        idx_tonkie = class_names.index(class_tonkie)
    except ValueError:
        return Response(f"Класс '{class_tonkie}' не найден в модели. Доступны: {class_names}", status_code=400)
    try:
        idx_ryadovye = class_names.index(class_ryadovye)
    except ValueError:
        return Response(f"Класс '{class_ryadovye}' не найден. Доступны: {class_names}", status_code=400)

    # Предобработка всего изображения (grayscale + фильтры)
    gray_processed = preprocess_image(img_bgr)

    # Дополнение до кратности BLOCK_SIZE (отражением)
    new_w = ((w + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    new_h = ((h + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    pad_right = new_w - w
    pad_bottom = new_h - h
    padded_gray = cv2.copyMakeBorder(gray_processed, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

    # Создаём копию исходного цветного изображения с дополнением (для рисования рамок)
    padded_color = cv2.copyMakeBorder(img_bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

    # Нарезка на блоки и классификация
    blocks_x = new_w // BLOCK_SIZE
    blocks_y = new_h // BLOCK_SIZE

    for i in range(blocks_y):
        for j in range(blocks_x):
            top = i * BLOCK_SIZE
            bottom = top + BLOCK_SIZE
            left = j * BLOCK_SIZE
            right = left + BLOCK_SIZE

            block = padded_gray[top:bottom, left:right]
            pred_idx = classify_block(block)

            # Рисуем рамку на дополненном цветном изображении
            if pred_idx == idx_tonkie:
                color = (0, 255, 255)   # жёлтый в BGR
            elif pred_idx == idx_ryadovye:
                color = (0, 255, 0)     # зелёный в BGR
            else:
                continue  # другие классы не обводим

            cv2.rectangle(padded_color, (left, top), (right-1, bottom-1), color, thickness)

    # Обрезаем дополнение и возвращаем результат
    result = padded_color[:h, :w]

    success, encoded = cv2.imencode('.png', result)
    if not success:
        return Response("Ошибка кодирования", status_code=500)

    return Response(content=encoded.tobytes(), media_type="image/png")