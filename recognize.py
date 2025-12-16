"""
Распознавание жестов в реальном времени с использованием MediaPipe
Основано на официальном репозитории: https://github.com/GibranBenitez/IPN-hand
"""
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp

from model import load_model
from dataset import IPNDataset

# MediaPipe для работы с руками
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def recognize_realtime(model_path: str = 'models/best_model.h5'):
    """Распознавание жестов в реальном времени с камеры"""
    
    # Загружаем модель
    if not Path(model_path).exists():
        print(f"Модель не найдена: {model_path}")
        print("Сначала обучите модель: python train.py")
        print("\nБыстрый старт:")
        print("  python train.py --max_train 10 --max_test 5 --epochs 10")
        return
    
    print("Загрузка модели...")
    model = load_model(model_path)
    print("Модель загружена!")
    
    dataset = IPNDataset()
    
    # MediaPipe для реального времени
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Открываем камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nРаспознавание запущено!")
    print("Нажмите 'q' для выхода")
    print("-" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Зеркально отражаем
        frame = cv2.flip(frame, 1)
        
        # Извлекаем landmarks с помощью MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # Отрисовываем скелет руки
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Извлекаем landmarks
            landmarks = np.zeros(63)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[idx * 3] = landmark.x
                    landmarks[idx * 3 + 1] = landmark.y
                    landmarks[idx * 3 + 2] = landmark.z
            
            # Нормализуем landmarks
            landmarks_normalized = dataset.normalize_landmarks(landmarks)
            
            # Подготавливаем данные для модели
            frame_input = np.expand_dims(landmarks_normalized, axis=0)
            
            # Предсказание
            predictions = model.predict(frame_input, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # Получаем название жеста
            gesture_name = dataset.CLASS_NAMES.get(predicted_class, "Unknown")
            
            # Отображаем результат
            text = f"{gesture_name} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, "Show your hand", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Отображаем кадр
        cv2.imshow('Hand Gesture Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nРаспознавание остановлено.")


if __name__ == '__main__':
    recognize_realtime()

