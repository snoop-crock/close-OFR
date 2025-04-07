import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import Config
from utils.logger import setup_logger
import warnings

# Настройка предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)


class DataProcessor:
    """Класс для обработки и подготовки данных с расширенной функциональностью"""

    def __init__(self, config):
        self.config = config
        self.cat_features = None  # Будет заполнено в preprocess_data
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Проверка существования и доступности файлов данных"""
        try:
            if not Path(self.config.INPUT_FILE).exists():
                raise FileNotFoundError(
                    f"Файл данных не найден: {self.config.INPUT_FILE}")
            if not os.access(self.config.INPUT_FILE, os.R_OK):
                raise PermissionError(
                    f"Нет доступа для чтения файла: {self.config.INPUT_FILE}")
        except Exception as e:
            logger.error(f"Ошибка проверки путей: {str(e)}")
            raise

    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """
        Загрузка и предварительная обработка данных
        Returns:
            Tuple: (X, y, cat_features)
        """
        # Загрузка данных
        X, y = self.load_data()

        # Предобработка
        X_clean, y_clean, cat_features = self.preprocess_data(X, y)

        return X_clean, y_clean, cat_features

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Загрузка данных из файла с валидацией

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Признаки и целевая переменная

        Raises:
            ValueError: Если целевая колонка отсутствует
            Exception: При ошибках чтения файла
        """
        logger.info(f"Загрузка данных из {self.config.INPUT_FILE}...")

        try:
            # Определение формата файла
            file_ext = Path(self.config.INPUT_FILE).suffix.lower()

            if file_ext == '.csv':
                data = pd.read_csv(self.config.INPUT_FILE)
            elif file_ext in ('.xlsx', '.xls'):
                data = pd.read_excel(self.config.INPUT_FILE)
            else:
                raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")

            # Проверка целевой переменной
            if self.config.TARGET_COL not in data.columns:
                raise ValueError(
                    f"Целевая колонка '{self.config.TARGET_COL}' не найдена. Доступные колонки: {list(data.columns)}")

            # Отделение признаков от целевой переменной
            X = data.drop(columns=[self.config.TARGET_COL])
            y = data[self.config.TARGET_COL]

            logger.info(
                f"Успешно загружено {len(X)} записей, {len(X.columns)} признаков")
            return X, y

        except Exception as e:
            logger.error(
                f"Критическая ошибка загрузки данных: {str(e)}", exc_info=True)
            raise

    def preprocess_data(self,
                        X: pd.DataFrame,
                        y: pd.Series,
                        handle_missing: str = 'drop',
                        cat_threshold: int = 10) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """
        Расширенная предобработка данных

        Args:
            X: DataFrame с признаками
            y: Series с целевой переменной
            handle_missing: Стратегия обработки пропусков ('drop', 'fill')
            cat_threshold: Порог для определения категориальных признаков

        Returns:
            Tuple: Очищенные данные, целевая переменная и индексы категориальных признаков
        """
        logger.info("Начало предобработки данных...")

        try:
            # Сохраняем исходное количество строк
            initial_rows = len(X)

            # Обработка бесконечных значений
            X = X.replace([np.inf, -np.inf], np.nan)

            # Обработка пропусков
            if handle_missing == 'drop':
                initial_len = len(X)
                X = X.dropna()
                y = y[X.index]
                if len(X) == 0:
                    raise ValueError(
                        "После удаления пропусков не осталось данных")
                dropped_rows = initial_rows - len(X)
                if dropped_rows > 0:
                    logger.warning(
                        f"Удалено {dropped_rows} записей с пропусками")
            elif handle_missing == 'fill':
                # Заполнение числовых и категориальных признаков
                for col in X.columns:
                    if pd.api.types.is_numeric_dtype(X[col]):
                        X[col] = X[col].fillna(X[col].median())
                    else:
                        X[col] = X[col].fillna('missing')

            # Определение категориальных признаков
            self.cat_features = []
            for col in X.columns:
                if (X[col].dtype == 'object') or (X[col].nunique() < cat_threshold):
                    # Стандартизация категориальных значений
                    X[col] = X[col].astype(str).str.strip(
                    ).str.lower().replace('nan', 'missing')
                    self.cat_features.append(col)

            logger.info(
                f"Обнаружено {len(self.cat_features)} категориальных признаков")
            return X, y, [X.columns.get_loc(c) for c in self.cat_features]

        except Exception as e:
            logger.error(
                f"Ошибка предобработки данных: {str(e)}", exc_info=True)
            raise

    def split_data(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   test_size: Optional[float] = None,
                   random_state: Optional[int] = None,
                   stratify: Optional[bool] = None) -> Tuple:
        """
        Улучшенное разделение данных на train/test

        Args:
            X: Признаки
            y: Целевая переменная
            test_size: Размер тестовой выборки
            random_state: Seed для воспроизводимости
            stratify: Стратифицированное разбиение

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Разделение данных на train/test...")

        try:
            test_size = test_size or self.config.TEST_SIZE
            random_state = random_state or self.config.RANDOM_STATE
            stratify = y if (stratify or self.config.STRATIFIED) else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )

            logger.info(
                f"Разделение завершено. Train: {len(X_train)} записей, "
                f"Test: {len(X_test)} записей"
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Ошибка разделения данных: {str(e)}", exc_info=True)
            raise

    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """
        Получение имен признаков с указанием их типа

        Args:
            X: DataFrame с признаками

        Returns:
            List[str]: Список имен признаков с указанием типа

        Raises:
            ValueError: Если категориальные признаки не определены
        """
        if self.cat_features is None:
            raise ValueError("Сначала необходимо выполнить preprocess_data()")

        return [
            f"{col} (cat)" if i in self.cat_features else f"{col} (num)"
            for i, col in enumerate(X.columns)
        ]

    def get_feature_stats(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Генерирует статистику по признакам

        Args:
            X: DataFrame с признаками

        Returns:
            pd.DataFrame: Статистика по признакам
        """
        stats = []
        for col in X.columns:
            if col in self.cat_features:
                stats.append({
                    'feature': col,
                    'type': 'categorical',
                    'unique': X[col].nunique(),
                    'missing': X[col].isnull().sum(),
                    'top_value': X[col].mode()[0] if len(X[col].mode()) > 0 else None
                })
            else:
                stats.append({
                    'feature': col,
                    'type': 'numerical',
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'missing': X[col].isnull().sum()
                })

        return pd.DataFrame(stats)
