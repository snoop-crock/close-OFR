import os
from pathlib import Path


class Config:
    """Конфигурация проекта с автоматическим созданием директорий и проверкой путей"""

    # Базовые пути
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    REPORTS_DIR = BASE_DIR / "reports"
    LOGS_DIR = BASE_DIR / "logs"

    # Файлы данных
    INPUT_FILE = DATA_DIR / "Normalize_avg.xlsx"

    # Настройки данных
    TARGET_COL = "OFR"
    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    STRATIFIED = False

    # Настройки модели
    N_TRIALS = 2
    GPU_PARAMS = {
        'task_type': "GPU",
        'devices': '0',
        'verbose': 100,
        'allow_writing_files': False
    }

    # Диапазоны параметров для Optuna
    PARAM_RANGES = {
        'iterations': (5000, 15000),
        'learning_rate': (1e-5, 0.3),
        'depth': (6, 12),
        'l2_leaf_reg': (1e-3, 10),
        'bagging_temperature': (0.0, 10.0),
        'random_strength': (1e-3, 10)
    }

    # Настройки кросс-валидации и анализа
    CV_FOLDS = 5
    CV_METRIC = 'RMSE'
    SHAP_SAMPLES = 100
    EARLY_STOPPING_ROUNDS = 50

    @classmethod
    def setup_dirs(cls):
        """Создает все необходимые директории и проверяет доступность файлов"""
        from utils.logger import setup_logger
        logger = setup_logger(__name__, log_dir=cls.LOGS_DIR)
        try:
            # Создаем основные директории
            cls.DATA_DIR.mkdir(exist_ok=True, parents=True)
            cls.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
            cls.LOGS_DIR.mkdir(exist_ok=True, parents=True)

            # Проверяем существование файла данных
            if not cls.INPUT_FILE.exists():
                raise FileNotFoundError(
                    f"Файл данных не найден: {cls.INPUT_FILE}\n"
                    f"Пожалуйста, поместите файл 'Normalize_avg.xlsx' в папку: {cls.DATA_DIR}"
                )

            # Проверяем доступность файла для чтения
            if not os.access(cls.INPUT_FILE, os.R_OK):
                raise PermissionError(
                    f"Нет доступа для чтения файла: {cls.INPUT_FILE}")

            logger.info(
                f"Конфигурация загружена. Данные будут браться из: {cls.INPUT_FILE}")

        except Exception as e:
            logger.error(f"Ошибка инициализации конфигурации: {str(e)}")
            raise

    @classmethod
    def validate_paths(cls):
        from utils.logger import setup_logger
        logger = setup_logger(__name__, log_dir=cls.LOGS_DIR)
        """Валидация всех путей в конфиге"""
        required_dirs = [cls.DATA_DIR, cls.REPORTS_DIR, cls.LOGS_DIR]
        for directory in required_dirs:
            if not directory.exists():
                logger.warning(f"Директория не существует: {directory}")

        if not cls.INPUT_FILE.exists():
            logger.error(f"Файл данных не найден: {cls.INPUT_FILE}")
            return False

        return True
