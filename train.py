"""
Простое обучение модели на датасете IPN Hand
Основано на официальном репозитории: https://github.com/GibranBenitez/IPN-hand
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from dataset import IPNDataset
from model import create_model, save_model


def prepare_data(dataset: IPNDataset, split: str = 'train', max_videos: int = None):
    """Подготавливает данные для обучения"""
    print(f"Загрузка аннотаций для {split}...")
    annotations = dataset.load_annotations(split)
    
    # Группируем по видео
    videos_dict = {}
    for ann in annotations:
        video_name = ann['video']
        if video_name not in videos_dict:
            videos_dict[video_name] = []
        videos_dict[video_name].append(ann)
    
    video_names = list(videos_dict.keys())
    if max_videos:
        video_names = video_names[:max_videos]
    
    print(f"Обработка {len(video_names)} видео...")
    
    X, y = [], []
    
    for video_name in tqdm(video_names, desc=f"Обработка {split}"):
        frames = dataset.load_video_frames(video_name)
        if len(frames) == 0:
            continue
        
        video_annotations = videos_dict[video_name]
        
        for frame_idx, frame in enumerate(frames):
            # Извлекаем landmarks с помощью MediaPipe
            landmarks = dataset.extract_landmarks(frame)
            
            # Пропускаем кадры без обнаруженной руки (только для жестов, не для non-gesture)
            label = dataset.get_frame_label(frame_idx, video_annotations)
            if label > 0 and np.all(landmarks == 0):
                continue  # Пропускаем кадры жестов без руки
            
            # Нормализуем landmarks
            landmarks_normalized = dataset.normalize_landmarks(landmarks)
            
            X.append(landmarks_normalized)
            y.append(label)
    
    return np.array(X), np.array(y)


def train(data_dir: str = 'data', max_train_videos: int = None, 
          max_test_videos: int = None, epochs: int = 50, batch_size: int = 32):
    """Обучает модель"""
    print("=" * 70)
    print("Обучение модели распознавания жестов IPN Hand")
    print("=" * 70)
    
    dataset = IPNDataset(data_dir)
    
    # Загружаем данные
    print("\n[1/3] Подготовка обучающих данных...")
    X_train, y_train = prepare_data(dataset, 'train', max_train_videos)
    
    print("\n[2/3] Подготовка тестовых данных...")
    X_test, y_test = prepare_data(dataset, 'test', max_test_videos)
    
    print(f"\nОбучающих образцов: {len(X_train)}")
    print(f"Тестовых образцов: {len(X_test)}")
    
    # Создаем модель
    print("\n[3/3] Создание и обучение модели...")
    model = create_model(input_shape=63, num_classes=14)  # 63 признака для MediaPipe landmarks
    model.summary()
    
    # Обучаем
    print(f"\nНачало обучения ({epochs} эпох)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Сохраняем модель
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'gesture_model.h5'
    save_model(model, str(model_path))
    
    # Оценка
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nТочность на тесте: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение модели IPN Hand')
    parser.add_argument('--data_dir', type=str, default='data', help='Путь к датасету')
    parser.add_argument('--max_train', type=int, default=None, help='Макс. видео для обучения')
    parser.add_argument('--max_test', type=int, default=None, help='Макс. видео для теста')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        max_train_videos=args.max_train,
        max_test_videos=args.max_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

