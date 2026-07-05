import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import HTMLResponse, Response
from pathlib import Path

# Импортируем ваши функции предобработки (они уже есть в share_photo_unet.py)
from shlif_nornik.share_photo_unet import (
    apply_clahe, apply_retinex,
    apply_bilateral_filter, apply_median_filter
)
from shlif_nornik.u_net import UNet

# ---------- Конфигурация ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'unet_best.pth'
BLOCK_SIZE = 512
IMG_SIZE = (512, 512)

# ---------- Загрузка модели ----------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print(f"Модель загружена на {DEVICE}")

# ---------- Функция предсказания блока ----------
def predict_block(block_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    block_rgb = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2RGB)
    block_resized = cv2.resize(block_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(block_resized).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float().squeeze().cpu().numpy()

    mask = (pred * 255).astype(np.uint8)
    h, w = block_bgr.shape[:2]
    if (h, w) != IMG_SIZE:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

# ---------- FastAPI приложение ----------
app = FastAPI(title="Segmentation (Talc → Red)")

HTML_FORM = """
<!DOCTYPE html>
<html>
<head><title>Сегментация талька</title></head>
<body>
    <h2>Загрузите изображение</h2>
    <form action="/predict/" method="post" enctype="multipart/form-data">
        <p>
            <label>Порог (0.1–0.9, чем выше, тем меньше выделений):</label>
            <input type="number" name="threshold" value="0.6" step="0.05" min="0.1" max="0.9">
        </p>
        <p>
            <label>Шумоподавление:</label>
            <select name="denoise">
                <option value="none">Выключено</option>
                <option value="bilateral" selected>Bilateral</option>
                <option value="median">Median</option>
            </select>
        </p>
        <p>
            <label>Коррекция освещения:</label>
            <select name="light">
                <option value="none" selected>Выключено</option>
                <option value="clahe">CLAHE</option>
                <option value="retinex">Retinex</option>
            </select>
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
    threshold: float = Form(0.6),
    denoise: str = Form("none"),
    light: str = Form("none")
):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return Response("Ошибка чтения изображения", status_code=400)

    h, w = img.shape[:2]
    new_w = ((w + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    new_h = ((h + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    pad_right = new_w - w
    pad_bottom = new_h - h
    padded = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

    full_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    blocks_x = new_w // BLOCK_SIZE
    blocks_y = new_h // BLOCK_SIZE

    # Определяем функции предобработки (если выбраны)
    denoise_func = None
    light_func = None

    if denoise == "bilateral":
        denoise_func = lambda x: apply_bilateral_filter(x, d=9, sigma_color=75, sigma_space=75)
    elif denoise == "median":
        denoise_func = lambda x: apply_median_filter(x, ksize=5)

    if light == "clahe":
        light_func = lambda x: apply_clahe(x, clip_limit=2.0, tile_grid_size=(8, 8))
    elif light == "retinex":
        light_func = lambda x: apply_retinex(x, sigma_list=(15, 80, 250))

    for i in range(blocks_y):
        for j in range(blocks_x):
            top = i * BLOCK_SIZE
            bottom = top + BLOCK_SIZE
            left = j * BLOCK_SIZE
            right = left + BLOCK_SIZE

            block = padded[top:bottom, left:right].copy()

            # Предобработка
            if denoise_func is not None:
                block = denoise_func(block)
            if light_func is not None:
                block = light_func(block)

            mask_block = predict_block(block, threshold)
            full_mask[top:bottom, left:right] = mask_block

    final_mask = full_mask[:h, :w]

    # Наложение красного цвета (BGR: 0,0,255)
    result = img.copy()
    result[final_mask == 255] = [0, 0, 255]

    success, encoded = cv2.imencode('.png', result)
    if not success:
        return Response("Ошибка кодирования", status_code=500)

    return Response(content=encoded.tobytes(), media_type="image/png")