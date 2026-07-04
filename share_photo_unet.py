import cv2
import numpy as np
import math
from pathlib import Path

# ==========================================
# 1. ФУНКЦИИ ИЗ ВАШИХ ФАЙЛОВ (light.py & border.py)
# ==========================================

def imread_unicode(filepath):
    """Читает изображение, поддерживает пути с кириллицей (возвращает BGR)."""
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return None

# --- Освещение (из light.py) ---
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_retinex(img, sigma_list=(15, 80, 250)):
    img_float = np.float64(img) + 1.0
    img_log = np.log(img_float)
    retinex = np.zeros_like(img_float)
    for sigma in sigma_list:
        L = cv2.GaussianBlur(img_float, (0, 0), sigma)
        retinex += img_log - np.log(L + 1.0)
    retinex /= len(sigma_list)
    retinex = np.exp(retinex)
    return np.clip(retinex, 0, 255).astype(np.uint8)

# --- Шумоподавление (из border.py) ---
def apply_median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)

def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


# ==========================================
# 2. ГЕНЕРАТОР БЛОКОВ С ПРЕДОБРАБОТКОЙ
# ==========================================

def split_and_preprocess(image_path, block_size=1000,
                         use_denoise=True, denoise_method='bilateral',
                         use_light=True, light_method='clahe'):
    """
    Делит изображение на блоки и применяет предобработку к каждому блоку.
    """
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение.")

    height, width = img.shape[:2]

    # Вычисляем размеры с padding (кратные block_size)
    new_width = math.ceil(width / block_size) * block_size
    new_height = math.ceil(height / block_size) * block_size

    # Добавляем поля методом ОТРАЖЕНИЯ (BORDER_REFLECT_101).
    # Это критически важно для фильтров, чтобы на краях блоков не было черных артефактов.
    pad_right = new_width - width
    pad_bottom = new_height - height
    padded_img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)

    num_blocks_x = new_width // block_size
    num_blocks_y = new_height // block_size

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            top = i * block_size
            bottom = top + block_size
            left = j * block_size
            right = left + block_size
            
            # Вырезаем блок (формат BGR)
            block = padded_img[top:bottom, left:right]

            # --- ЭТАП 1: Шумоподавление ---
            if use_denoise:
                if denoise_method == 'bilateral':
                    block = apply_bilateral_filter(block, d=9, sigma_color=75, sigma_space=75)
                elif denoise_method == 'median':
                    block = apply_median_filter(block, ksize=5)

            # --- ЭТАП 2: Нормализация освещения ---
            if use_light:
                if light_method == 'clahe':
                    block = apply_clahe(block, clip_limit=2.0, tile_grid_size=(8, 8))
                elif light_method == 'retinex':
                    block = apply_retinex(block, sigma_list=(15, 80, 250))

            # Отдаем готовый блок в нейросеть
            yield block


# ==========================================
# 3. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ==========================================
if __name__ == "__main__":
    image_file = 'C://Users/User/Desktop/hak/Панорамы/4.jpg' # Путь к вашей большой фотографии
    
    # Настройки предобработки
    DENOISE = True
    DENOISE_METHOD = 'bilateral' # 'bilateral' или 'median'
    
    LIGHT = True
    LIGHT_METHOD = 'clahe'       # 'clahe' или 'retinex'

    print("Начинаем нарезку и предобработку...")
    
    # Получаем генератор
    blocks_gen = split_and_preprocess(
        image_file, 
        block_size=1000,
        use_denoise=DENOISE, denoise_method=DENOISE_METHOD,
        use_light=LIGHT, light_method=LIGHT_METHOD
    )
    
    for idx, block in enumerate(blocks_gen):
        print(f"Блок {idx + 1}: форма {block.shape}, тип {block.dtype}")
        
        # ==========================================
        # ЗДЕСЬ ПЕРЕДАЧА В НЕЙРОСЕТЬ
        # ==========================================
        # Внимание: OpenCV читает в BGR, а нейросети (PyTorch/TensorFlow) 
        # обычно ожидают RGB. Нужно поменять каналы местами:
        block_rgb = cv2.cvtColor(block, cv2.COLOR_BGR2RGB)
        
        # Далее стандартная конвертация в тензор (пример для PyTorch):
        # import torch
        # import torchvision.transforms as T
        # transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # tensor_block = transform(block_rgb).unsqueeze(0)
        
        # prediction = model(tensor_block)