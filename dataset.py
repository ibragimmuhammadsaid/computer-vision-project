"""
Упрощенная загрузка датасета IPN Hand с использованием MediaPipe
Основано на официальном репозитории:
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import mediapipe as mp

# MediaPipe для извлечения landmarks руки
mp_hands = mp.solutions.hands


class IPNDataset:
    """Простой класс для работы с датасетом IPN Hand"""
    
    # Маппинг классов жестов
    GESTURE_CLASSES = {
        'D0X': 0,  # Non-gesture
        'B0A': 1,  # Pointing with one finger
        'B0B': 2,  # Pointing with two fingers
        'G01': 3,  # Click with one finger
        'G02': 4,  # Click with two fingers
        'G03': 5,  # Throw up
        'G04': 6,  # Throw down
        'G05': 7,  # Throw left
        'G06': 8,  # Throw right
        'G07': 9,  # Open twice
        'G08': 10, # Double click with one finger
        'G09': 11, # Double click with two fingers
        'G10': 12, # Zoom in
        'G11': 13, # Zoom out
    }
    
    CLASS_NAMES = {
        0: 'Non-gesture',
        1: 'Point 1 finger',
        2: 'Point 2 fingers',
        3: 'Click 1 finger',
        4: 'Click 2 fingers',
        5: 'Throw up',
        6: 'Throw down',
        7: 'Throw left',
        8: 'Throw right',
        9: 'Open twice',
        10: 'Double click 1',
        11: 'Double click 2',
        12: 'Zoom in',
        13: 'Zoom out',
    }
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.annotations_dir = self.data_dir / 'annotations'
        # MediaPipe для извлечения landmarks
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    def load_annotations(self, split: str = 'train') -> List[dict]:
        """
        Загружает аннотации для train или test
        
        Формат: video_name,label,?,start_frame,end_frame,duration
        """
        if split == 'train':
            annot_file = self.annotations_dir / 'Annot_TrainList.txt'
        else:
            annot_file = self.annotations_dir / 'Annot_TestList.txt'
        
        annotations = []
        with open(annot_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) >= 6:
                    video_name = parts[0]
                    label = parts[1]
                    start_frame = int(parts[3]) - 1  # 0-based
                    end_frame = int(parts[4]) - 1
                    
                    annotations.append({
                        'video': video_name,
                        'label': label,
                        'start': start_frame,
                        'end': end_frame
                    })
        
        return annotations
    
    def get_video_frames_path(self, video_name: str) -> Path:
        """Находит путь к кадрам видео"""
        for frames_dir in ['frames01', 'frames02', 'frames03', 'frames04', 'frames05']:
            frames_path = self.data_dir / 'frames' / frames_dir / video_name
            if frames_path.exists():
                return frames_path
        return None
    
    def extract_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        Извлекает landmarks руки из изображения
        
        Returns:
            Массив из 63 значений (x, y, z для 21 точки)
            Если рука не обнаружена, возвращает нулевой массив
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks = np.zeros(63)  # 21 точка * 3 координаты
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmarks[idx * 3] = landmark.x
                landmarks[idx * 3 + 1] = landmark.y
                landmarks[idx * 3 + 2] = landmark.z
        
        return landmarks
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Нормализует landmarks относительно запястья"""
        landmarks = landmarks.copy()
        wrist = landmarks[:3]
        
        for i in range(0, len(landmarks), 3):
            landmarks[i] -= wrist[0]  # x
            landmarks[i + 1] -= wrist[1]  # y
            landmarks[i + 2] -= wrist[2]  # z
        
        return landmarks
    
    def load_video_frames(self, video_name: str) -> List[np.ndarray]:
        """Загружает все кадры видео"""
        frames_path = self.get_video_frames_path(video_name)
        if frames_path is None:
            return []
        
        frames = []
        frame_files = sorted(frames_path.glob('*.jpg'))
        
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def get_frame_label(self, frame_idx: int, annotations: List[dict]) -> int:
        """Получает метку для кадра"""
        for ann in annotations:
            if ann['start'] <= frame_idx <= ann['end']:
                return self.GESTURE_CLASSES.get(ann['label'], 0)
        return 0  # Non-gesture по умолчанию

