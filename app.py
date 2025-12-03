import sys
import warnings

from PyQt5 import QtWidgets

from anpr.ui.main_window import MainWindow
from logging_manager import LoggingManager, get_logger
from settings_manager import SettingsManager

# Silence noisy quantization warnings emitted by torch on repeated startups.
warnings.filterwarnings(
    "ignore",
    message="Please use quant_min and quant_max to specify the range for observers.",
    module="torch.ao.quantization.observer",
)
warnings.filterwarnings(
    "ignore",
    message="must run observer before calling calculate_qparams",
    module="torch.ao.quantization.observer",
)

logger = get_logger(__name__)


def main() -> None:
    """Entrypoint that wires settings, logging and the main window."""

    settings = SettingsManager()
    LoggingManager(settings.get_logging_config())
    logger.info("Запуск ANPR Desktop")

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(settings)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
