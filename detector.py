import argparse
import os
import time
from typing import List, Dict, Any, Tuple
import logging

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import QConfigMapping
import numpy as np

from logging_manager import LoggingManager, get_logger



class Config:
    """Класс для хранения всех конфигурационных параметров."""
    YOLO_MODEL_PATH: str = 'models/yolo/best.pt'
    OCR_MODEL_PATH: str = 'models/ocr_crnn/cnn.pth'
    
    OCR_IMG_HEIGHT: int = 32
    OCR_IMG_WIDTH: int = 128
    OCR_ALPHABET: str = '0123456789ABCEHKMOPTXY'
    OCR_CONFIDENCE_THRESHOLD: float = 0.6

    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5

    TRACK_BEST_SHOTS: int = 3

    DEVICE: torch.device = torch.device("cpu")


logger = get_logger(__name__)


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # --- CNN часть (Глаза) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2), # -> height: 16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d(2, 2), # -> height: 8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d((2, 1), (2, 1)), # -> height: 4
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.ReLU(True), 
            nn.MaxPool2d((2, 1), (2, 1)) # -> height: 2. ЭТОГО СЛОЯ НЕ БЫЛО, ДОБАВЛЯЕМ ЕГО!
        )
        
        # --- RNN часть (Мозг) ---
        
        self.rnn = nn.LSTM(512 * 2, 256, bidirectional=True, num_layers=2, batch_first=True)
        
        # --- Classifier (Рот) ---
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Прогоняем через CNN
        x = self.cnn(x) # -> (batch, 512, 2, 32)
        
        
        # "Распрямляем" выход CNN для подачи в RNN
        # объединяем каналы и высоту
        batch, channels, height, width = x.size()
        # Заменяем .view() на .reshape() для большей надежности
        x = x.reshape(batch, channels * height, width) 
        
        # Меняем оси местами для RNN, который ожидает (batch, seq_len, features)
        x = x.permute(0, 2, 1) # -> (batch, 32, 1024)
        
        # Прогоняем через RNN
        x, _ = self.rnn(x) # -> (batch, 32, 512)
        
        # Прогоняем через классификатор
        x = self.classifier(x) # -> (batch, 32, num_classes)
        
        # Для CTCLoss нам нужен формат (sequence_length, batch, num_classes)
        x = x.permute(1, 0, 2) # -> (32, batch, num_classes)
        x = nn.functional.log_softmax(x, dim=2)
        
        return x


class YOLODetector:
    """Обертка для модели детекции YOLO."""

    def __init__(self, model_path: str, device: torch.device):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device
        self._tracking_supported = True
        logger.info("Детектор YOLO успешно загружен (model=%s, device=%s)", model_path, device)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Обнаруживает номера на ОДНОМ кадре (для изображений)."""
        detections = self.model.predict(frame, verbose=False, device=self.device)
        results = []
        for det in detections[0].boxes.data:
            x1, y1, x2, y2, conf, _ = det.cpu().numpy()
            if conf >= Config.DETECTION_CONFIDENCE_THRESHOLD:
                results.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(conf)})
        return results

    def _track_internal(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        detections = self.model.track(frame, persist=True, verbose=False, device=self.device)
        results: List[Dict[str, Any]] = []
        if detections[0].boxes.id is None:
            return results

        track_ids = detections[0].boxes.id.int().cpu().tolist()
        boxes = detections[0].boxes.xyxy.cpu().numpy()
        confs = detections[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            if conf >= Config.DETECTION_CONFIDENCE_THRESHOLD:
                results.append(
                    {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "confidence": float(conf),
                        "track_id": track_id,
                    }
                )
        return results

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Отслеживает номера в последовательности кадров (для видео) с откатом к детекции."""
        if not self._tracking_supported:
            return self.detect(frame)

        try:
            return self._track_internal(frame)
        except ModuleNotFoundError:
            # Отсутствующие зависимости трекера (например, lap/byte-track) ломают поток при повторном
            # запуске. Запоминаем, что трекинг недоступен, и продолжаем с обычной детекцией.
            self._tracking_supported = False
            logger.warning("Отключаем трекинг YOLO: отсутствуют зависимости")
            return self.detect(frame)
        except Exception:
            # Любые другие ошибки трекера (например, при одновременном запуске нескольких каналов)
            # не должны приводить к падению — отключаем трекинг и продолжаем детекцию.
            self._tracking_supported = False
            logger.exception("Отключаем трекинг YOLO из-за ошибки, переключаемся на detect")
            return self.detect(frame)
    # Ниже оставлены только методы с откатом на детекцию, чтобы избежать падений из-за
    # необязательных зависимостей трекера.


class CRNNRecognizer:
    """Обертка для квантованной модели распознавания CRNN."""
    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Grayscale(),
            transforms.Resize((Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH)),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.int_to_char = {i + 1: char for i, char in enumerate(Config.OCR_ALPHABET)}
        self.int_to_char[0] = '' # CTC Blank token
        
        
        num_classes = len(Config.OCR_ALPHABET) + 1
        
        # 1. Создаем "пустой" скелет модели и переводим в режим инференса
        model_to_load = CRNN(num_classes).eval()
        
        # 2. Готовим его к квантизации точно так же, как при сохранении
        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.get_default_qconfig('fbgemm'))
        example_inputs = (torch.randn(1, 1, Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH),)
        model_prepared = quantize_fx.prepare_fx(model_to_load, qconfig_mapping, example_inputs)
        model_quantized = quantize_fx.convert_fx(model_prepared)
        
        # 3. И только теперь загружаем сохраненные веса
        model_quantized.load_state_dict(torch.load(model_path, map_location=device))
        self.model = model_quantized
        logger.info("Распознаватель OCR (INT8) успешно загружен (model=%s, device=%s)", model_path, device)

    @torch.no_grad()
    def recognize(self, plate_image: np.ndarray) -> tuple[str, float]:
        preprocessed_plate = self.transform(plate_image).unsqueeze(0).to(self.device)
        preds = self.model(preprocessed_plate)
        return self._decode_with_confidence(preds)

    def _decode_with_confidence(self, log_probs: torch.Tensor) -> tuple[str, float]:
        """Декодирует CTC-выход и возвращает текст и уверенность (0..1)."""
        # log_probs: (seq_len, batch, num_classes)
        probs = log_probs.permute(1, 0, 2)[0]  # (seq_len, num_classes)
        time_steps = probs.size(0)

        decoded_chars: list[str] = []
        char_confidences: list[float] = []
        last_char_idx = 0

        for t in range(time_steps):
            timestep_log_probs = probs[t]
            char_idx = int(torch.argmax(timestep_log_probs).item())
            char_conf = float(torch.exp(torch.max(timestep_log_probs)).item())

            if char_idx != 0 and char_idx != last_char_idx:
                decoded_chars.append(self.int_to_char.get(char_idx, ''))
                char_confidences.append(char_conf)

            last_char_idx = char_idx

        text = "".join(decoded_chars)
        if not char_confidences:
            return text, 0.0

        # Усредняем уверенность по символам, чтобы штрафовать длинные шумные последовательности.
        avg_confidence = sum(char_confidences) / len(char_confidences)
        return text, avg_confidence

class Visualizer:
    """Отвечает за отрисовку результатов."""
    @staticmethod
    def draw_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        for res in results:
            x1, y1, x2, y2 = res['bbox']
            text = res.get('text', '')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame


from collections import Counter

class TrackAggregator:
    """Агрегирует результаты распознавания в рамках одного трека."""

    def __init__(self, best_shots: int):
        self.best_shots = max(1, best_shots)
        self.track_texts: Dict[int, List[str]] = {}
        self.last_emitted: Dict[int, str] = {}

    def add_result(self, track_id: int, text: str) -> str:
        """Сохраняет промежуточный результат и возвращает консенсус, если он сформирован."""
        if not text:
            return ""

        bucket = self.track_texts.setdefault(track_id, [])
        bucket.append(text)
        if len(bucket) > self.best_shots:
            bucket.pop(0)

        counts = Counter(bucket)
        consensus, freq = counts.most_common(1)[0]
        # Консенсус выдается, когда собраны заданное число бестшотов и есть большинство.
        quorum = max(1, (self.best_shots + 1) // 2)
        has_quorum = len(bucket) >= self.best_shots and freq >= quorum
        if has_quorum and self.last_emitted.get(track_id) != consensus:
            self.last_emitted[track_id] = consensus
            return consensus
        return ""


class ANPR_Pipeline:
    """Главный класс, управляющий процессом распознавания."""

    def __init__(
        self,
        recognizer: CRNNRecognizer,
        best_shots: int,
        cooldown_seconds: int = 0,
        min_confidence: float = Config.OCR_CONFIDENCE_THRESHOLD,
    ):
        self.recognizer = recognizer
        self.aggregator = TrackAggregator(best_shots)
        self.cooldown_seconds = max(0, cooldown_seconds)
        self.min_confidence = max(0.0, min(1.0, min_confidence))
        self._last_seen: Dict[str, float] = {}

    def _on_cooldown(self, plate: str) -> bool:
        last_seen = self._last_seen.get(plate)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < self.cooldown_seconds

    def _touch_plate(self, plate: str) -> None:
        self._last_seen[plate] = time.monotonic()

    # --- МЕТОДЫ ДЛЯ КОРРЕКЦИИ ПЕРСПЕКТИВЫ ---
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <= 0: return image
        dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return plate_image
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                return self._four_point_transform(plate_image, approx.reshape(4, 2))
        return plate_image

    # --- ГЛАВНЫЙ МЕТОД ОБРАБОТКИ ---
    def process_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # 1. УЛУЧШАЕМ ПРЕПРОЦЕССИНГ
                processed_plate = self._preprocess_plate(roi)
                
                if processed_plate.size > 0:
                    # 2. РАСПОЗНАЕМ
                    current_text, confidence = self.recognizer.recognize(processed_plate)

                    if confidence < self.min_confidence:
                        detection['text'] = "Нечитаемо"
                        detection['unreadable'] = True
                        detection['confidence'] = confidence
                        continue

                    # 3. Агрегируем по треку или фиксируем напрямую для одиночных фото
                    if 'track_id' in detection:
                        detection['text'] = self.aggregator.add_result(
                            detection['track_id'], current_text
                        )
                    else:  # Для одиночных фото
                        detection['text'] = current_text

                    detection['confidence'] = confidence

                    if self.cooldown_seconds > 0 and detection.get('text'):
                        if self._on_cooldown(detection['text']):
                            detection['text'] = ""
                        else:
                            self._touch_plate(detection['text'])
        return detections

def process_source(pipeline: ANPR_Pipeline, detector: YOLODetector, source_path: str):
    """Обрабатывает источник, выбирая нужный метод (detect/track)."""
    is_video = source_path.endswith(('.mp4', '.avi', '.mov')) or source_path.isnumeric()
    
    if is_video:
        cap = cv2.VideoCapture(int(source_path) if source_path.isnumeric() else source_path)
        if not cap.isOpened(): raise IOError(f"Ошибка...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # --- ИСПОЛЬЗУЕМ ТРЕКИНГ ДЛЯ ВИДЕО ---
            detections = detector.track(frame)
            results = pipeline.process_frame(frame, detections) # Передаем detections в пайплайн
            
            frame = Visualizer.draw_results(frame, results)
            cv2.imshow("ANPR Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(source_path)
        if frame is None: raise IOError(f"Ошибка...")
        
        # --- ИСПОЛЬЗУЕМ ДЕТЕКЦИЮ ДЛЯ ФОТО ---
        detections = detector.detect(frame)
        results = pipeline.process_frame(frame, detections) # Передаем detections в пайплайн
        
        print(f"\nНа изображении '{os.path.basename(source_path)}' распознаны номера:")
        if not results: print("- Номера не найдены.")
        for res in results: print(f"- {res['text']} (уверенность детектора: {res['confidence']:.2f})")
        frame = Visualizer.draw_results(frame, results)
        cv2.imshow("ANPR Result", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    """Главная функция, точка входа в программу."""
    parser = argparse.ArgumentParser(description="Распознавание автомобильных номеров.")
    parser.add_argument("--source", required=True, help="Путь к изображению, видеофайлу или ID веб-камеры (напр., '0').")
    args = parser.parse_args()

    try:
        LoggingManager()
        detector = YOLODetector(Config.YOLO_MODEL_PATH, Config.DEVICE)
        recognizer = CRNNRecognizer(Config.OCR_MODEL_PATH, Config.DEVICE)
        pipeline = ANPR_Pipeline(recognizer, Config.TRACK_BEST_SHOTS)
        process_source(pipeline, detector, args.source)
    except (IOError, FileNotFoundError) as e:
        print(f"Критическая ошибка: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")

if __name__ == '__main__':
    main()
