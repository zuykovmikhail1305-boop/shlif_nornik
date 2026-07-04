import os
import logging
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

# ВАЖНО: класс CNN1000 должен быть определён здесь точно так же, как при обучении!
# (скопируйте из вашего обучающего скрипта)
class CNN1000(torch.nn.Module):
    # ... (весь код класса из вашего скрипта) ...
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CNN Grayscale Classifier API")

MODEL_PATH = os.getenv("MODEL_PATH", "grayscale_1000_cnn_pytorch.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели и метаданных при старте
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = CNN1000(num_classes=checkpoint['num_classes']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    CLASS_NAMES = checkpoint['class_names']
    IMG_SIZE = checkpoint['img_size']
    
    logger.info(f"Модель загружена. Классы: {CLASS_NAMES}, размер: {IMG_SIZE}")
except Exception as e:
    logger.critical(f"Ошибка загрузки модели: {e}")
    model = None

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {
        "status": "healthy",
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Принимает изображение (PNG/JPG) и возвращает предсказание"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        # Читаем файл из запроса
        image = Image.open(file.file).convert('L')  # grayscale
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(prob, 1)
            
            all_probs = {
                CLASS_NAMES[i]: prob[0][i].item()
                for i in range(len(CLASS_NAMES))
            }
        
        return {
            "predicted_class": CLASS_NAMES[pred_idx.item()],
            "confidence": conf.item(),
            "all_probabilities": all_probs,
        }
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)