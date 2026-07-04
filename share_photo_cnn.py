import cv2
import numpy as np
import math

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
    if len(img.shape) == 2:  # Поддержка Grayscale
        return clahe.apply(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def apply_retinex(img, sigma_list=(15, 80, 250)):
    # Примечание: Retinex лучше работает с цветом. Для grayscale используйте CLAHE.
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
# 2. ГЕНЕРАТОР БЛОКОВ С ПРЕДОБРАБОТКОЙ И GRAYSCALE
# ==========================================

def split_and_preprocess(image_path, block_size=1000,
                         to_grayscale=True,  # <-- НОВЫЙ ПАРАМЕТР
                         use_denoise=True, denoise_method='bilateral',
                         use_light=True, light_method='clahe'):
    """
    Делит изображение на блоки, переводит в grayscale (опционально) 
    и применяет предобработку к каждому блоку.
    """
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение.")

    height, width = img.shape[:2]

    # Вычисляем размеры с padding
    new_width = math.ceil(width / block_size) * block_size
    new_height = math.ceil(height / block_size) * block_size

    # Добавляем поля методом отражения
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
            
            # Вырезаем блок (изначально BGR, 3 канала)
            block = padded_img[top:bottom, left:right]

            # --- ЭТАП 0: Перевод в Grayscale ---
            if to_grayscale:
                block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
                # Теперь block имеет форму (1000, 1000) вместо (1000, 1000, 3)

            # --- ЭТАП 1: Шумоподавление ---
            if use_denoise:
                if denoise_method == 'bilateral':
                    block = apply_bilateral_filter(block, d=9, sigma_color=75, sigma_space=75)
                elif denoise_method == 'median':
                    block = apply_median_filter(block, ksize=5)

            # --- ЭТАП 2: Нормализация освещения ---
            if use_light:
                if light_method == 'clahe':
                    # CLAHE отлично работает с grayscale благодаря проверке len(img.shape) == 2
                    block = apply_clahe(block, clip_limit=2.0, tile_grid_size=(8, 8))
                elif light_method == 'retinex':
                    block = apply_retinex(block, sigma_list=(15, 80, 250))

            yield block


# ==========================================
# 3. ПРИМЕР ИСПОЛЬЗОВАНИЯ И ПЕРЕДАЧИ В НЕЙРОСЕТЬ
# ==========================================
if __name__ == "__main__":
    image_file = "your_large_photo.jpg" 
    
    # Настройки
    TO_GRAYSCALE = True
    DENOISE_METHOD = 'bilateral' 
    LIGHT_METHOD = 'clahe'       

    print("Начинаем нарезку, перевод в grayscale и предобработку...")
    
    blocks_gen = split_and_preprocess(
        image_file, 
        block_size=1000,
        to_grayscale=TO_GRAYSCALE,
        use_denoise=True, denoise_method=DENOISE_METHOD,
        use_light=True, light_method=LIGHT_METHOD
    )
    
    for idx, block in enumerate(blocks_gen):
        print(f"Блок {idx + 1}: форма {block.shape}") 
        # Если to_grayscale=True, форма будет (1000, 1000)
        # Если to_grayscale=False, форма будет (1000, 1000, 3)
        
        # ==========================================
        # ПОДГОТОВКА ТЕНЗОРА ДЛЯ НЕЙРОСЕТИ (PyTorch)
        # ==========================================
        import torch
        import torchvision.transforms as T

        # 1. Нормализация значений в диапазон [0, 1]
        tensor_block = torch.from_numpy(block).float() / 255.0
        
        # 2. Изменение размерности для PyTorch
        if TO_GRAYSCALE:
            # Для grayscale: (H, W) -> (1, H, W)
            tensor_block = tensor_block.unsqueeze(0) 
            
            # ВАЖНО: Если ваша нейросеть обучалась на цветных изображениях (3 канала),
            # ей нужно подать 3 канала. В таком случае продублируйте канал:
            # tensor_block = tensor_block.repeat(3, 1, 1) 
        else:
            # Для цвета: (H, W, C) -> (C, H, W)
            tensor_block = tensor_block.permute(2, 0, 1)
            
        # 3. Добавление размерности batch_size: (C, H, W) -> (1, C, H, W)
        tensor_block = tensor_block.unsqueeze(0)
        
        # 4. Стандартная нормализация (опционально, зависит от модели)
        # Если модель обучена на ImageNet:
        # normalize = T.Normalize(mean=[0.485], std=[0.229]) # Для 1 канала
        # tensor_block = normalize(tensor_block)
        
        # Теперь tensor_block готов к подаче в модель:
        # prediction = model(tensor_block)