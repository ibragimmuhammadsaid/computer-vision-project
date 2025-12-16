"""
Простая модель для распознавания жестов рук на основе MediaPipe landmarks
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(input_shape: int = 63, num_classes: int = 14):
    """
    Создает простую модель для распознавания жестов на основе MediaPipe landmarks
    
    Args:
        input_shape: Количество признаков (63 для MediaPipe - 21 точка * 3 координаты)
        num_classes: Количество классов жестов (14)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_model(model_path: str):
    """Загружает сохраненную модель"""
    return keras.models.load_model(model_path)


def save_model(model: keras.Model, model_path: str):
    """Сохраняет модель"""
    model.save(model_path)
    print(f"Модель сохранена: {model_path}")

