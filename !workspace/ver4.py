"""
ГЛАВНЫЙ СКРИПТ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ CATBOOST С АВТОМАТИЧЕСКОЙ ОПТИМИЗАЦИЕЙ ГИПЕРПАРАМЕТРОВ

Архитектурные улучшения:
1. Разделение на модули (классы) по функциональности
2. Четкое разделение конфигурации и логики
3. Улучшенная обработка ошибок и логирование
4. Гибкость конфигурации через классы
5. Документирование всех компонентов
"""

import os
import logging
from typing import Dict, Tuple, List, Optional, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from scipy.stats import spearmanr
from catboost import CatBoostRegressor, Pool, utils as catboost_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Класс конфигурации для всех параметров проекта"""

    # Пути и файлы
    INPUT_FILE = "Normalize_avg.xlsx"
    OUTPUT_FOLDER = "model_results"
    TARGET_COL = "OFR"

    # Настройки обучения
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_TRIALS = 2

    # Настройки GPU
    GPU_CONFIG = {
        'task_type': "GPU",
        'devices': '0',
        'boost_from_average': False,
        'allow_writing_files': False,
        'verbose': 100
    }

    # Диапазоны параметров для Optuna
    PARAM_RANGES = {
        'iterations': (1000, 10000),
        'learning_rate': (1e-4, 0.3),
        'depth': (4, 10),
        'l2_leaf_reg': (1e-3, 10),
        'bagging_temperature': (0.0, 10.0),
        'random_strength': (1e-3, 10)
    }

    # Параметры логирования
    LOGGING_CONFIG = {
        'log_file': 'training.log',
        'log_level': logging.INFO
    }


class DataProcessor:
    """Класс для обработки и подготовки данных"""

    def __init__(self, config: Config):
        self.config = config
        self._ensure_output_folder()

    def _ensure_output_folder(self) -> None:
        """Создает папку для результатов если ее нет"""
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, pd.Series, List[int]]:
        """Загрузка и предварительная обработка данных"""
        logger.info("Загрузка и предобработка данных")

        try:
            data = pd.read_excel(self.config.INPUT_FILE)

            if self.config.TARGET_COL not in data.columns:
                raise ValueError(
                    f"Целевая переменная {self.config.TARGET_COL} не найдена")

            X = data.drop(columns=[self.config.TARGET_COL])
            y = data[self.config.TARGET_COL]

            # Обработка пропущенных значений
            X = X.dropna()
            y = y[X.index]

            # Обработка категориальных признаков
            cat_features = self._process_categorical_features(X)

            logger.info(
                f"Данные загружены. Признаков: {X.shape[1]}, Примеров: {X.shape[0]}")
            logger.info(
                f"Обнаружено категориальных признаков: {len(cat_features)}")

            return X, y, cat_features

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise

    def _process_categorical_features(self, X: pd.DataFrame) -> List[int]:
        """Обработка категориальных признаков"""
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].nunique() < 10:
                X[col] = X[col].astype(str).replace('nan', 'missing')
                cat_features.append(col)
        return [X.columns.get_loc(c) for c in cat_features]

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Разделение данных на train/validation"""
        return train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )


class CatBoostTrainer:
    """Класс для обучения и оценки модели CatBoost"""

    def __init__(self, config: Config):
        self.config = config

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Dict[str, Any],
        cat_features: List[int]
    ) -> CatBoostRegressor:
        """Обучение модели CatBoost"""
        logger.info("Начало обучения модели")

        try:
            logger.info("Параметры модели:")
            for param, value in params.items():
                logger.info(f"{param}: {value}")

            model = CatBoostRegressor(**params)

            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)

            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                verbose=params.get('verbose', False)
            )
            logger.info("Обучение модели завершено")
            return model

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            raise

    def evaluate(self, model: CatBoostRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Оценка качества модели"""
        logger.info("Оценка качества модели")

        y_pred = model.predict(X)

        metrics = {
            'MSE': mean_squared_error(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred),
            'Spearman': spearmanr(y, y_pred)[0]
        }

        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        return metrics


class HyperparameterOptimizer:
    """Класс для оптимизации гиперпараметров с помощью Optuna"""

    def __init__(self, config: Config, trainer: CatBoostTrainer):
        self.config = config
        self.trainer = trainer

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: List[int],
        gpu_params: Dict[str, Any],
        n_trials: int
    ) -> Tuple[Dict[str, Any], optuna.study.Study]:
        """Оптимизация гиперпараметров"""
        logger.info("Начало оптимизации гиперпараметров")

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.config.RANDOM_STATE))

        objective = self._create_objective(
            X_train, y_train, X_val, y_val, cat_features, gpu_params)

        study.optimize(
            objective,
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=True
        )

        logger.info("Оптимизация завершена. Лучшие параметры:")
        for param, value in study.best_params.items():
            logger.info(f"{param}: {value}")

        return study.best_params, study

    def _create_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_features: List[int],
        gpu_params: Dict[str, Any]
    ):
        """Создание функции цели для Optuna"""
        def objective(trial: optuna.Trial) -> float:
            params = {
                'iterations': trial.suggest_int('iterations', *self.config.PARAM_RANGES['iterations']),
                'learning_rate': trial.suggest_float('learning_rate', *self.config.PARAM_RANGES['learning_rate'], log=True),
                'depth': trial.suggest_int('depth', *self.config.PARAM_RANGES['depth']),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *self.config.PARAM_RANGES['l2_leaf_reg'], log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', *self.config.PARAM_RANGES['bagging_temperature']),
                'random_strength': trial.suggest_float('random_strength', *self.config.PARAM_RANGES['random_strength']),
                **gpu_params
            }

            model = self.trainer.train(
                X_train, y_train, X_val, y_val, params, cat_features)
            return mean_squared_error(y_val, model.predict(X_val))

        return objective


class ModelAnalyzer:
    """Класс для анализа и визуализации результатов модели"""

    def __init__(self, config: Config):
        self.config = config

    def analyze_features(
        self,
        model: CatBoostRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: List[int]
    ) -> pd.DataFrame:
        """Анализ важности признаков"""
        logger.info("Анализ важности признаков")

        perm_importance = permutation_importance(
            model, X, y, n_repeats=5, random_state=self.config.RANDOM_STATE)

        model_importance = model.get_feature_importance()

        feature_analysis = pd.DataFrame({
            'Feature': X.columns,
            'Permutation_Importance': perm_importance.importances_mean,
            'Model_Importance': model_importance,
            'Type': ['Categorical' if i in cat_features else 'Numerical' for i in range(len(X.columns))]
        }).sort_values('Permutation_Importance', ascending=False)

        logger.info("Топ-10 важных признаков:")
        logger.info(feature_analysis.head(10).to_string())

        return feature_analysis

    def save_visualizations(
        self,
        model: CatBoostRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        output_folder: str
    ) -> None:
        """Сохранение визуализаций"""
        logger.info("Сохранение визуализаций")

        # True vs Predicted
        self._save_true_vs_predicted_plot(model, X, y, output_folder)

        # Correlation matrix
        self._save_correlation_matrix(X, output_folder)

    def _save_true_vs_predicted_plot(
        self,
        model: CatBoostRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        output_folder: str
    ) -> None:
        """Сохранение графика истинных vs предсказанных значений"""
        plt.figure(figsize=(10, 6))
        y_pred = model.predict(X)
        sns.regplot(x=y, y=y_pred, scatter_kws={'alpha': 0.3})
        plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values')
        plt.savefig(os.path.join(output_folder, 'true_vs_pred.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _save_correlation_matrix(self, X: pd.DataFrame, output_folder: str) -> None:
        """Сохранение матрицы корреляций"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig(os.path.join(output_folder, 'correlation_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


class ResultSaver:
    """Класс для сохранения результатов обучения"""

    def __init__(self, config: Config):
        self.config = config

    def save_all(
        self,
        model: CatBoostRegressor,
        study: optuna.study.Study,
        feature_analysis: pd.DataFrame,
        metrics: Dict[str, float],
        output_folder: str
    ) -> None:
        """Сохранение всех результатов"""
        logger.info("Сохранение результатов обучения")

        self._save_model(model, output_folder)
        self._save_study(study, output_folder)
        self._save_report(study, feature_analysis, metrics, output_folder)

        logger.info(f"Все результаты сохранены в папку: {output_folder}")

    def _save_model(self, model: CatBoostRegressor, output_folder: str) -> None:
        """Сохранение модели"""
        model_path = os.path.join(output_folder, 'model.cbm')
        model.save_model(model_path)
        logger.info(f"Модель сохранена: {model_path}")

    def _save_study(self, study: optuna.study.Study, output_folder: str) -> None:
        """Сохранение исследования Optuna"""
        study_path = os.path.join(output_folder, 'optuna_study.pkl')
        joblib.dump(study, study_path)
        logger.info(f"Исследование Optuna сохранено: {study_path}")

    def _save_report(
        self,
        study: optuna.study.Study,
        feature_analysis: pd.DataFrame,
        metrics: Dict[str, float],
        output_folder: str
    ) -> None:
        """Создание полного отчета"""
        report_path = os.path.join(output_folder, 'report.xlsx')

        with pd.ExcelWriter(report_path) as writer:
            pd.DataFrame([metrics]).to_excel(
                writer, sheet_name='Metrics', index=False)
            feature_analysis.to_excel(
                writer, sheet_name='Feature_Importance', index=False)
            study.trials_dataframe().to_excel(
                writer, sheet_name='Optimization_History', index=False)
            pd.DataFrame([study.best_params]).to_excel(
                writer, sheet_name='Best_Params', index=False)

        logger.info(f"Полный отчет сохранен: {report_path}")


class ModelTrainingPipeline:
    """Основной класс для выполнения всего пайплайна обучения"""

    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.trainer = CatBoostTrainer(config)
        self.optimizer = HyperparameterOptimizer(config, self.trainer)
        self.analyzer = ModelAnalyzer(config)
        self.saver = ResultSaver(config)

    def run(self) -> None:
        """Запуск полного пайплайна обучения"""
        try:
            logger.info("Запуск пайплайна обучения модели")

            # 2. Загрузка и подготовка данных
            X, y, cat_features = self.data_processor.load_and_preprocess()
            X_train, X_val, y_train, y_val = self.data_processor.split_data(
                X, y)

            # 3. Оптимизация гиперпараметров
            best_params, study = self.optimizer.optimize(
                X_train, y_train, X_val, y_val,
                cat_features, self.config.N_TRIALS
            )

            # 4. Обучение финальной модели
            final_model = self.trainer.train(
                X_train, y_train, X_val, y_val,
                {**best_params}, cat_features
            )

            # 5. Оценка модели
            val_metrics = self.trainer.evaluate(final_model, X_val, y_val)

            # 6. Анализ признаков
            feature_analysis = self.analyzer.analyze_features(
                final_model, X_val, y_val, cat_features)

            # 7. Визуализация
            self.analyzer.save_visualizations(
                final_model, X_train, y_train, self.config.OUTPUT_FOLDER)

            # 8. Сохранение результатов
            self.saver.save_all(
                final_model, study, feature_analysis,
                val_metrics, self.config.OUTPUT_FOLDER
            )

            logger.info("Пайплайн обучения успешно завершен")

        except Exception as e:
            logger.error(f"Ошибка в пайплайне обучения: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        config = Config()
        pipeline = ModelTrainingPipeline(config)
        pipeline.run()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {str(e)}")
        raise
