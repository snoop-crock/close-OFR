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

            # Конвертируем целевую переменную в числовой тип
            y = pd.to_numeric(data[self.config.TARGET_COL], errors='coerce')

            # Обрабатываем NaN значения после конвертации
            if y.isnull().any():
                num_nans = y.isnull().sum()
                logger.warning(
                    f"Обнаружено {num_nans} NaN значений в целевой переменной после конвертации")
                X = X[~y.isnull()]
                y = y.dropna()

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
        Полная предварительная обработка данных с расширенными проверками

        Args:
            X: DataFrame с признаками
            y: Series с целевой переменной
            handle_missing: Стратегия обработки пропусков ('drop', 'fill', 'raise')
            cat_threshold: Порог уникальных значений для определения категориальных признаков

        Returns:
            Tuple: Очищенные данные (X, y) и индексы категориальных признаков

        Raises:
            ValueError: При проблемах с данными или параметрами
        """
        logger.info("Начало комплексной предобработки данных...")
        logger.info(f"Исходная форма данных: X={X.shape}, y={y.shape}")

        try:
            # 1. Проверка входных данных
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X должен быть pandas DataFrame")
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise TypeError("y должен быть pandas Series или numpy array")
            if len(X) != len(y):
                raise ValueError("X и y должны иметь одинаковую длину")

            # 2. Обработка целевой переменной
            y_clean = pd.to_numeric(y, errors='coerce')
            if y_clean.isnull().any():
                num_nans = y_clean.isnull().sum()
                logger.warning(
                    f"Обнаружено {num_nans} NaN в целевой переменной после конвертации")

                if handle_missing == 'raise':
                    raise ValueError(
                        f"Найдено {num_nans} NaN в целевой переменной")
                elif handle_missing == 'drop':
                    valid_idx = y_clean.notnull()
                    X_clean = X[valid_idx].copy()
                    y_clean = y_clean[valid_idx].copy()
                    logger.info(
                        f"Удалено {num_nans} строк с NaN в целевой переменной")
                    if len(X_clean) == 0:
                        raise ValueError(
                            "После обработки NaN не осталось данных")
                else:
                    raise ValueError(
                        f"Неизвестный метод обработки пропусков: {handle_missing}")

            # 3. Обработка бесконечных значений в признаках
            X_clean = X.copy() if 'X_clean' not in locals() else X_clean
            X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

            # 4. Обработка пропусков в признаках
            if X_clean.isnull().any().any():
                missing_cols = X_clean.columns[X_clean.isnull().any()].tolist()
                num_missing = X_clean.isnull().sum().sum()
                logger.warning(
                    f"Обнаружено {num_missing} пропусков в признаках: {missing_cols}")

                if handle_missing == 'drop':
                    X_clean = X_clean.dropna()
                    y_clean = y_clean[X_clean.index]
                    logger.info(f"Удалено {num_missing} строк с пропусками")
                elif handle_missing == 'fill':
                    for col in X_clean.columns:
                        if pd.api.types.is_numeric_dtype(X_clean[col]):
                            fill_val = X_clean[col].median()
                            X_clean[col] = X_clean[col].fillna(fill_val)
                            logger.debug(
                                f"Заполнены пропуски в {col} медианой: {fill_val}")
                        else:
                            X_clean[col] = X_clean[col].fillna('missing')
                            logger.debug(
                                f"Заполнены пропуски в {col} строкой 'missing'")
                else:
                    raise ValueError(
                        f"Неизвестный метод обработки пропусков: {handle_missing}")

            # 5. Определение категориальных признаков
            self.cat_features = []
            cat_indices = []

            for col in X_clean.columns:
                # Признак считается категориальным если:
                # 1) Имеет тип object
                # 2) Имеет меньше cat_threshold уникальных значений
                # 3) Или явно указан как категориальный
                is_categorical = (
                    X_clean[col].dtype == 'object' or
                    X_clean[col].nunique() < cat_threshold or
                    (hasattr(self, 'force_categorical')
                     and col in self.force_categorical)
                )

                if is_categorical:
                    # Стандартизация категориальных значений
                    X_clean[col] = (
                        X_clean[col]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .replace(['nan', 'none', 'null'], 'missing')
                    )
                    self.cat_features.append(col)
                    cat_indices.append(X_clean.columns.get_loc(col))
                    logger.debug(f"Признак {col} обработан как категориальный")

            logger.info(
                f"Обнаружено {len(self.cat_features)} категориальных признаков: {self.cat_features}")

            # 6. Финализация данных
            if len(X_clean) == 0:
                raise ValueError("После обработки не осталось данных")

            logger.info(
                f"Форма данных после обработки: X={X_clean.shape}, y={y_clean.shape}")
            logger.info(f"Типы признаков:\n{X_clean.dtypes}")

            return X_clean, y_clean, cat_indices

        except Exception as e:
            logger.error(
                f"Критическая ошибка предобработки данных: {str(e)}", exc_info=True)
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
