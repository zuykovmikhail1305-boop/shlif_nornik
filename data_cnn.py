import os
import cv2
import numpy as np
import math
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
# Папка с исходным датасетом (структура: raw_data/class_name/image.jpg)
RAW_DATA_DIR = 'C://Users/User/Desktop/hak/photo' 
# Папка, куда сохранятся предобработанные данные (для обучения)
PROCESSED_DATA_DIR = 'C://Users/User/Desktop/hak/train_cnn_2' 

BLOCK_SIZE = 512  # Размер блока (должен совпадать с IMG_SIZE в модели)
USE_DENOISE = True
USE_LIGHT = True
# ================================================

# --- Вспомогательные функции для работы с кириллицей ---
def imread_unicode(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return None

def imwrite_unicode(filepath, img):
    ext = os.path.splitext(filepath)[1]
    is_success, buffer = cv2.imencode(ext, img)
    if is_success:
        with open(filepath, 'wb') as f:
            f.write(buffer)

# --- Функции предобработки ---
def apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def preprocess_and_split(img, block_size=512):
    # 1. Перевод в Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Шумоподавление
    if USE_DENOISE:
        img = apply_bilateral_filter(img)
        
    # 3. Нормализация освещения (CLAHE)
    if USE_LIGHT:
        img = apply_clahe(img)
        
    # 4. Нарезка на блоки (с паддингом, если размер не кратен block_size)
    h, w = img.shape
    new_h = math.ceil(h / block_size) * block_size
    new_w = math.ceil(w / block_size) * block_size
    
    pad_bottom = new_h - h
    pad_right = new_w - w
    padded_img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101)
    
    blocks = []
    for i in range(0, new_h, block_size):
        for j in range(0, new_w, block_size):
            block = padded_img[i:i+block_size, j:j+block_size]
            blocks.append(block)
            
    return blocks

# ==================== ОСНОВНОЙ ЦИКЛ ====================
if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Папка {RAW_DATA_DIR} не найдена!")
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    print(f"Найдено классов: {len(classes)}")
    
    for class_name in classes:
        src_class_dir = os.path.join(RAW_DATA_DIR, class_name)
        dst_class_dir = os.path.join(PROCESSED_DATA_DIR, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\nОбработка класса: '{class_name}' ({len(images)} изображений)")
        
        for img_name in tqdm(images, desc=class_name):
            img_path = os.path.join(src_class_dir, img_name)
            img = imread_unicode(img_path)
            
            if img is None:
                continue
                
            blocks = preprocess_and_split(img, BLOCK_SIZE)
            
            # Сохраняем каждый блок как отдельное изображение
            base_name = os.path.splitext(img_name)[0]
            for idx, block in enumerate(blocks):
                save_name = f"{base_name}_block_{idx}.png"
                save_path = os.path.join(dst_class_dir, save_name)
                imwrite_unicode(save_path, block)
                
    print(f"\n✅ Предобработка завершена! Данные сохранены в: {PROCESSED_DATA_DIR}")