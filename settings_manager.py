import json
import os
from typing import Any, Dict, List


class SettingsManager:
    """Управляет конфигурацией приложения и каналами."""

    def __init__(self, path: str = "settings.json") -> None:
        self.path = path
        self.settings = self._load()

    def _default(self) -> Dict[str, Any]:
        return {
            "grid": "2x2",
            "channels": [
                {"id": 1, "name": "Канал 1", "source": "0"},
            ],
            "storage": {"events_db": "data/events.db"},
            "tracking": {"best_shots": 3},
        }

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            defaults = self._default()
            self._save(defaults)
            return defaults
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_channels(self) -> List[Dict[str, Any]]:
        return self.settings.get("channels", [])

    def save_channels(self, channels: List[Dict[str, Any]]) -> None:
        self.settings["channels"] = channels
        self._save(self.settings)

    def get_grid(self) -> str:
        return self.settings.get("grid", "2x2")

    def save_grid(self, grid: str) -> None:
        self.settings["grid"] = grid
        self._save(self.settings)

    def get_db_path(self) -> str:
        storage = self.settings.get("storage", {})
        return storage.get("events_db", "data/events.db")

    def get_best_shots(self) -> int:
        tracking = self.settings.get("tracking", {})
        return int(tracking.get("best_shots", 3))

    def save_best_shots(self, best_shots: int) -> None:
        tracking = self.settings.get("tracking", {})
        tracking["best_shots"] = int(best_shots)
        self.settings["tracking"] = tracking
        self._save(self.settings)

    def refresh(self) -> None:
        self.settings = self._load()

    def update_channel(self, channel_id: int, data: Dict[str, Any]) -> None:
        channels = self.get_channels()
        for idx, channel in enumerate(channels):
            if channel.get("id") == channel_id:
                channels[idx].update(data)
                break
        else:
            channels.append(data)
        self.save_channels(channels)
