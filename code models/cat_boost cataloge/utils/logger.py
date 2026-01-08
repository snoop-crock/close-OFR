import logging
from pathlib import Path


def setup_logger(name: str, log_dir: Path = None) -> logging.Logger:
    """Настройка логгера"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Создаем директорию для логов, если ее нет

    if log_dir is None:
        log_dir = Path('logs')

    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Логи в файл
    file_handler = logging.FileHandler(
        log_dir / 'model_training.log',
        encoding='utf-8'
    )

    file_handler.setFormatter(formatter)

    # Вывод в консоль
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
