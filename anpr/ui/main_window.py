from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from anpr.workers.channel_worker import ChannelWorker
from logging_manager import get_logger
from settings_manager import SettingsManager
from storage import EventDatabase

logger = get_logger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ANPR с вкладками мониторинга, событий, поиска и настроек."""

    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3"]

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)

        self.settings = settings or SettingsManager()
        self.db = EventDatabase(self.settings.get_db_path())

        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, QtWidgets.QLabel] = {}

        self.tabs = QtWidgets.QTabWidget()
        self.monitor_tab = self._build_monitor_tab()
        self.events_tab = self._build_events_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.monitor_tab, "Монитор")
        self.tabs.addTab(self.events_tab, "События")
        self.tabs.addTab(self.search_tab, "Поиск")
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.setCentralWidget(self.tabs)
        self._refresh_events_table()

    # ------------------ Мониторинг ------------------
    def _build_monitor_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Сетка:"))
        self.grid_selector = QtWidgets.QComboBox()
        self.grid_selector.addItems(self.GRID_VARIANTS)
        self.grid_selector.setCurrentText(self.settings.get_grid())
        self.grid_selector.currentTextChanged.connect(self._on_grid_changed)
        controls.addWidget(self.grid_selector)

        self.start_button = QtWidgets.QPushButton("Запустить")
        self.start_button.clicked.connect(self._start_channels)
        controls.addWidget(self.start_button)

        controls.addStretch()
        controls.addWidget(QtWidgets.QLabel("Последнее событие:"))
        self.last_event_label = QtWidgets.QLabel("—")
        controls.addWidget(self.last_event_label)

        layout.addLayout(controls)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        layout.addWidget(self.grid_widget)

        self._draw_grid()
        return widget

    @staticmethod
    def _prepare_optional_datetime(widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setCalendarPopup(True)
        widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        min_dt = QtCore.QDateTime.fromSecsSinceEpoch(0)
        widget.setMinimumDateTime(min_dt)
        widget.setSpecialValueText("Не выбрано")
        widget.setDateTime(min_dt)

    @staticmethod
    def _get_datetime_value(widget: QtWidgets.QDateTimeEdit) -> Optional[str]:
        if widget.dateTime() == widget.minimumDateTime():
            return None
        return widget.dateTime().toString(QtCore.Qt.ISODate)

    def _draw_grid(self) -> None:
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.channel_labels.clear()
        channels = self.settings.get_channels()
        rows, cols = map(int, self.grid_selector.currentText().split("x"))
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = QtWidgets.QLabel("Нет сигнала")
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setStyleSheet(
                    "background-color: #1c1c1c; color: #ccc; border: 1px solid #444; padding: 4px;"
                )
                label.setMinimumSize(220, 170)
                label.setScaledContents(False)
                label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    label.setText(channel_name)
                    self.channel_labels[channel_name] = label
                self.grid_layout.addWidget(label, row, col)
                index += 1

    def _on_grid_changed(self, grid: str) -> None:
        self.settings.save_grid(grid)
        self._draw_grid()

    def _start_channels(self) -> None:
        self._stop_workers()
        self.channel_workers = []
        best_shots = self.settings.get_best_shots()
        cooldown = self.settings.get_cooldown_seconds()
        min_confidence = self.settings.get_min_confidence()
        max_fps = self.settings.get_max_fps()
        for channel_conf in self.settings.get_channels():
            worker = ChannelWorker(
                channel_conf,
                self.settings.get_db_path(),
                best_shots,
                cooldown,
                min_confidence,
                max_fps,
            )
            worker.frame_ready.connect(self._update_frame)
            worker.event_ready.connect(self._handle_event)
            worker.status_ready.connect(self._handle_status)
            self.channel_workers.append(worker)
            worker.start()

    def _stop_workers(self) -> None:
        for worker in self.channel_workers:
            worker.stop()
            worker.wait(1000)
        self.channel_workers = []

    def _update_frame(self, channel_name: str, image: QtGui.QImage) -> None:
        label = self.channel_labels.get(channel_name)
        if not label:
            return
        target_size = label.contentsRect().size()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(pixmap)

    def _handle_event(self, event: Dict) -> None:
        self.last_event_label.setText(
            f"{event['timestamp']} | {event['channel']} | {event['plate']} | {event['confidence']:.2f}"
        )
        self._refresh_events_table()

    def _handle_status(self, channel: str, status: str) -> None:
        label = self.channel_labels.get(channel)
        if label:
            label.setText(f"{channel}: {status}")

    # ------------------ События ------------------
    def _build_events_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(QtWidgets.QLabel("Дата с:"))
        self.events_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.events_from)
        filters.addWidget(self.events_from)

        filters.addWidget(QtWidgets.QLabel("по:"))
        self.events_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.events_to)
        filters.addWidget(self.events_to)

        filters.addWidget(QtWidgets.QLabel("Канал:"))
        self.events_channel = QtWidgets.QComboBox()
        self.events_channel.addItem("Все", "")
        for channel in self.settings.get_channels():
            self.events_channel.addItem(channel.get("name", ""), channel.get("name", ""))
        filters.addWidget(self.events_channel)

        filters.addWidget(QtWidgets.QLabel("Список номеров (через запятую):"))
        self.events_plate_list = QtWidgets.QLineEdit()
        filters.addWidget(self.events_plate_list)

        apply_btn = QtWidgets.QPushButton("Применить")
        apply_btn.clicked.connect(self._refresh_events_table)
        filters.addWidget(apply_btn)

        layout.addLayout(filters)

        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(
            ["Время", "Канал", "Номер", "Уверенность", "Источник"]
        )
        self.events_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.events_table)

        return widget

    def _refresh_events_table(self) -> None:
        start = self._get_datetime_value(self.events_from)
        end = self._get_datetime_value(self.events_to)
        channel = self.events_channel.currentData() if hasattr(self, "events_channel") else None
        plates_input = self.events_plate_list.text() if hasattr(self, "events_plate_list") else ""
        plates = [plate.strip() for plate in plates_input.split(",") if plate.strip()]

        rows = self.db.fetch_filtered(start=start or None, end=end or None, channel=channel or None, plates=plates)
        self.events_table.setRowCount(0)
        for row_data in rows:
            row_index = self.events_table.rowCount()
            self.events_table.insertRow(row_index)
            self.events_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row_data["timestamp"]))
            self.events_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            self.events_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.events_table.setItem(
                row_index, 3, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.events_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["source"]))

    # ------------------ Поиск ------------------
    def _build_search_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.search_plate = QtWidgets.QLineEdit()
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)

        form.addRow("Номер:", self.search_plate)
        form.addRow("Дата с:", self.search_from)
        form.addRow("Дата по:", self.search_to)
        layout.addLayout(form)

        search_btn = QtWidgets.QPushButton("Искать")
        search_btn.clicked.connect(self._run_plate_search)
        layout.addWidget(search_btn)

        self.search_table = QtWidgets.QTableWidget(0, 5)
        self.search_table.setHorizontalHeaderLabels(
            ["Время", "Канал", "Номер", "Уверенность", "Источник"]
        )
        self.search_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.search_table)

        return widget

    def _run_plate_search(self) -> None:
        start = self._get_datetime_value(self.search_from)
        end = self._get_datetime_value(self.search_to)
        plate_fragment = self.search_plate.text()
        rows = self.db.search_by_plate(plate_fragment, start=start or None, end=end or None)
        self.search_table.setRowCount(0)
        for row_data in rows:
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            self.search_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row_data["timestamp"]))
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            self.search_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 3, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["source"]))

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.currentRowChanged.connect(self._load_channel_form)
        layout.addWidget(self.channels_list)

        form_layout = QtWidgets.QFormLayout()
        self.best_shots_input = QtWidgets.QSpinBox()
        self.best_shots_input.setRange(1, 50)
        self.best_shots_input.setValue(self.settings.get_best_shots())
        self.best_shots_input.setToolTip("Количество бестшотов, участвующих в консенсусе трека")
        form_layout.addRow("Бестшоты на трек:", self.best_shots_input)
        self.cooldown_input = QtWidgets.QSpinBox()
        self.cooldown_input.setRange(0, 3600)
        self.cooldown_input.setValue(self.settings.get_cooldown_seconds())
        self.cooldown_input.setToolTip(
            "Интервал (в секундах), в течение которого не создается повторное событие для того же номера"
        )
        form_layout.addRow("Пауза повтора (сек):", self.cooldown_input)
        self.max_fps_input = QtWidgets.QSpinBox()
        self.max_fps_input.setRange(1, 60)
        self.max_fps_input.setValue(self.settings.get_max_fps())
        self.max_fps_input.setToolTip("Ограничение кадров в секунду для снижения нагрузки на ЦП")
        form_layout.addRow("Макс FPS каналов:", self.max_fps_input)
        self.min_conf_input = QtWidgets.QDoubleSpinBox()
        self.min_conf_input.setRange(0.0, 1.0)
        self.min_conf_input.setSingleStep(0.05)
        self.min_conf_input.setDecimals(2)
        self.min_conf_input.setValue(self.settings.get_min_confidence())
        self.min_conf_input.setToolTip(
            "Минимальная уверенность OCR (0-1) для приема результата; ниже — помечается как нечитаемое"
        )
        form_layout.addRow("Мин. уверенность OCR:", self.min_conf_input)
        self.channel_name_input = QtWidgets.QLineEdit()
        self.channel_source_input = QtWidgets.QLineEdit()
        form_layout.addRow("Название:", self.channel_name_input)
        form_layout.addRow("Источник/RTSP:", self.channel_source_input)

        buttons_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Добавить")
        add_btn.clicked.connect(self._add_channel)
        remove_btn = QtWidgets.QPushButton("Удалить")
        remove_btn.clicked.connect(self._remove_channel)
        save_btn = QtWidgets.QPushButton("Сохранить")
        save_btn.clicked.connect(self._save_channel)
        buttons_layout.addWidget(add_btn)
        buttons_layout.addWidget(remove_btn)
        buttons_layout.addWidget(save_btn)

        form_layout.addRow(buttons_layout)
        layout.addLayout(form_layout)

        self._reload_channels_list()
        return widget

    def _reload_channels_list(self) -> None:
        self.channels_list.clear()
        for channel in self.settings.get_channels():
            self.channels_list.addItem(channel.get("name", "Канал"))
        if self.channels_list.count():
            self.channels_list.setCurrentRow(0)

    def _load_channel_form(self, index: int) -> None:
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channel = channels[index]
            self.channel_name_input.setText(channel.get("name", ""))
            self.channel_source_input.setText(channel.get("source", ""))

    def _add_channel(self) -> None:
        channels = self.settings.get_channels()
        new_id = max([c.get("id", 0) for c in channels] + [0]) + 1
        channels.append({"id": new_id, "name": f"Канал {new_id}", "source": ""})
        self.settings.save_channels(channels)
        self._reload_channels_list()
        self._draw_grid()

    def _remove_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels.pop(index)
            self.settings.save_channels(channels)
            self._reload_channels_list()
            self._draw_grid()

    def _save_channel(self) -> None:
        self.settings.save_best_shots(self.best_shots_input.value())
        self.settings.save_cooldown_seconds(self.cooldown_input.value())
        self.settings.save_min_confidence(self.min_conf_input.value())
        self.settings.save_max_fps(self.max_fps_input.value())
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels[index]["name"] = self.channel_name_input.text()
            channels[index]["source"] = self.channel_source_input.text()
            self.settings.save_channels(channels)
            self._reload_channels_list()
            self._draw_grid()

    # ------------------ Жизненный цикл ------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_workers()
        event.accept()
