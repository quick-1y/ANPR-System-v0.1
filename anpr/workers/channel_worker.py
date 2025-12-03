from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui

from detector import ANPR_Pipeline, CRNNRecognizer, YOLODetector, Config as ModelConfig
from logging_manager import get_logger
from storage import EventDatabase

logger = get_logger(__name__)


class ChannelWorker(QtCore.QThread):
    """Background worker that captures frames, runs ANPR pipeline and emits UI events."""

    frame_ready = QtCore.pyqtSignal(str, QtGui.QImage)
    event_ready = QtCore.pyqtSignal(dict)
    status_ready = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        channel_conf: Dict,
        db_path: str,
        best_shots: int,
        cooldown_seconds: int,
        min_confidence: float,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.channel_conf = channel_conf
        self.db_path = db_path
        self._running = True
        self.best_shots = best_shots
        self.cooldown_seconds = cooldown_seconds
        self.min_confidence = min_confidence

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not capture.isOpened():
            return None
        return capture

    def _build_pipeline(self) -> Tuple[ANPR_Pipeline, YOLODetector]:
        detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
        recognizer = CRNNRecognizer(ModelConfig.OCR_MODEL_PATH, ModelConfig.DEVICE)
        return (
            ANPR_Pipeline(
                recognizer,
                self.best_shots,
                self.cooldown_seconds,
                min_confidence=self.min_confidence,
            ),
            detector,
        )

    def run(self) -> None:
        try:
            pipeline, detector = self._build_pipeline()
            storage = EventDatabase(self.db_path)

            source = str(self.channel_conf.get("source", "0"))
            capture = self._open_capture(source)
            if capture is None:
                self.status_ready.emit(self.channel_conf.get("name", "Канал"), "Нет сигнала")
                logger.warning("Не удалось открыть источник %s для канала %s", source, self.channel_conf)
                return

            logger.info("Канал %s запущен (источник=%s)", self.channel_conf.get("name", "Канал"), source)
            while self._running:
                ret, frame = capture.read()
                if not ret:
                    self.status_ready.emit(self.channel_conf.get("name", "Канал"), "Поток остановлен")
                    logger.warning("Поток остановлен для канала %s", self.channel_conf.get("name", "Канал"))
                    break

                detections = detector.track(frame)
                results = pipeline.process_frame(frame, detections)

                for res in results:
                    if res.get("unreadable"):
                        logger.debug(
                            "Канал %s: номер помечен как нечитаемый (confidence=%.2f)",
                            self.channel_conf.get("name", "Канал"),
                            res.get("confidence", 0.0),
                        )
                        continue
                    if res.get("text"):
                        event = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "channel": self.channel_conf.get("name", "Канал"),
                            "plate": res.get("text", ""),
                            "confidence": res.get("confidence", 0.0),
                            "source": source,
                        }
                        event["id"] = storage.insert_event(
                            channel=event["channel"],
                            plate=event["plate"],
                            confidence=event["confidence"],
                            source=event["source"],
                            timestamp=event["timestamp"],
                        )
                        self.event_ready.emit(event)
                        logger.info(
                            "Канал %s: зафиксирован номер %s (conf=%.2f, track=%s)",
                            event["channel"],
                            event["plate"],
                            event["confidence"],
                            res.get("track_id", "-"),
                        )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                # Копируем буфер, чтобы предотвратить обращение Qt к уже освобожденной памяти
                # во время перерисовок окна.
                q_image = QtGui.QImage(
                    rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
                ).copy()
                self.frame_ready.emit(self.channel_conf.get("name", "Канал"), q_image)

            capture.release()
        except Exception as exc:  # noqa: BLE001
            self.status_ready.emit(self.channel_conf.get("name", "Канал"), f"Ошибка: {exc}")
            logger.exception("Канал %s аварийно остановлен", self.channel_conf.get("name", "Канал"))

    def stop(self) -> None:
        self._running = False
