from catboost import CatBoostRegressor, Pool
import optuna
from typing import Dict, Any, Tuple, List, Union
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error
from scipy.stats import spearmanr
import shap
import pandas as pd
from config import Config
from utils.logger import setup_logger
import warnings

# Настройка предупреждений
warnings.filterwarnings('ignore', category=UserWarning)
logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)


class ModelTrainer:
    """Класс для обучения и оптимизации моделей CatBoost с расширенной функциональностью"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.best_model = None
        self.study = None
        self.shap_values = None
        self.shap_data = None

    def _create_pool(self,
                     X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     cat_features: List[int] = None) -> Pool:
        """
        Создает объект Pool для CatBoost

        Args:
            X: Матрица признаков
            y: Вектор целевой переменной
            cat_features: Список индексов категориальных признаков

        Returns:
            Объект Pool для CatBoost
        """
        try:
            # Исправлено: проверяем, есть ли у X атрибут columns (для DataFrame)
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                # Создаем имена признаков по умолчанию
                feature_names = [f'f{i}' for i in range(X.shape[1])]

            return Pool(
                data=X,
                label=y,
                cat_features=cat_features,
                feature_names=feature_names
            )
        except Exception as e:
            logger.error(f"Ошибка создания Pool: {str(e)}")
            raise

    def train_model(self,
                    params: Dict[str, Any],
                    X_train: Union[pd.DataFrame, np.ndarray],
                    y_train: Union[pd.Series, np.ndarray],
                    X_val: Union[pd.DataFrame, np.ndarray] = None,
                    y_val: Union[pd.Series, np.ndarray] = None,
                    cat_features: List[int] = None) -> CatBoostRegressor:
        """
        Обучает модель CatBoost с заданными параметрами

        Args:
            params: Словарь параметров модели
            X_train: Обучающая выборка
            y_train: Целевая переменная обучающей выборки
            X_val: Валидационная выборка (опционально)
            y_val: Целевая переменная валидационной выборки (опционально)
            cat_features: Список индексов категориальных признаков

        Returns:
            Обученная модель CatBoostRegressor
        """
        logger.info("Начало обучения модели CatBoost...")

        try:
            # Создаем Pool для обучения
            train_pool = self._create_pool(X_train, y_train, cat_features)
            eval_pool = None

            if X_val is not None and y_val is not None:
                eval_pool = self._create_pool(X_val, y_val, cat_features)
                logger.info(
                    f"Используется валидационный набор: {len(X_val)} образцов")

            # Инициализация модели
            model = CatBoostRegressor(
                **params,
                random_seed=self.config.RANDOM_STATE,
                train_dir=self.config.REPORTS_DIR / 'catboost_info'
            )

            # Обучение модели
            model.fit(
                train_pool,
                eval_set=eval_pool,
                use_best_model=eval_pool is not None,
                verbose=params.get('verbose', False),
                plot=False
            )

            self.best_model = model
            logger.info("Обучение модели успешно завершено")
            logger.info(f"Лучшая итерация: {model.get_best_iteration()}")
            logger.info(f"Лучший score: {model.get_best_score()}")

            return model

        except Exception as e:
            logger.error(
                f"Ошибка при обучении модели: {str(e)}", exc_info=True)
            raise

    def train_final_model(self,
                          best_params: Dict[str, Any],
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame = None,
                          y_val: pd.Series = None,
                          cat_features: List[int] = None) -> CatBoostRegressor:
        """
        Обучает финальную модель с лучшими параметрами

        Args:
            best_params: Оптимизированные параметры модели
            X_train: Обучающие данные
            y_train: Целевая переменная обучающих данных
            X_val: Валидационные данные (опционально)
            y_val: Целевая переменная валидационных данных (опционально)
            cat_features: Список индексов категориальных признаков

        Returns:
            Обученная модель CatBoostRegressor
        """
        logger.info("Обучение финальной модели...")

        # Объединяем лучшие параметры с GPU-параметрами
        final_params = best_params.copy()
        final_params.update(self.config.GPU_PARAMS)

        # Обучаем модель
        return self.train_model(
            params=final_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features
        )

    def optimize_hyperparameters(self,
                                 X_train: pd.DataFrame,
                                 y_train: pd.Series,
                                 X_val: pd.DataFrame,
                                 y_val: pd.Series,
                                 cat_features: List[int] = None,
                                 n_trials: int = None) -> Tuple[Dict[str, Any], optuna.study.Study]:
        """
        Оптимизирует гиперпараметры модели с помощью Optuna

        Args:
            X_train: Обучающая выборка
            y_train: Целевая переменная обучающей выборки
            X_val: Валидационная выборка
            y_val: Целевая переменная валидационной выборки
            cat_features: Список индексов категориальных признаков
            n_trials: Количество испытаний для Optuna

        Returns:
            Кортеж (лучшие параметры, объект Study)
        """
        logger.info("Начало оптимизации гиперпараметров с помощью Optuna...")

        n_trials = self.config.N_TRIALS
        train_pool = self._create_pool(X_train, y_train, cat_features)
        eval_pool = self._create_pool(X_val, y_val, cat_features)

        def objective(trial: optuna.Trial) -> float:
            """Целевая функция для Optuna"""
            params = {
                'iterations': trial.suggest_int('iterations', *self.config.PARAM_RANGES['iterations']),
                'learning_rate': trial.suggest_float('learning_rate', *self.config.PARAM_RANGES['learning_rate'], log=True),
                'depth': trial.suggest_int('depth', *self.config.PARAM_RANGES['depth']),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *self.config.PARAM_RANGES['l2_leaf_reg'], log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', *self.config.PARAM_RANGES['bagging_temperature']),
                'random_strength': trial.suggest_float('random_strength', *self.config.PARAM_RANGES['random_strength']),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'verbose': False,
                **self.config.GPU_PARAMS
            }

            try:
                model = CatBoostRegressor(**params)
                model.fit(train_pool, eval_set=eval_pool,
                          use_best_model=True, verbose=False)

                # Используем RMSE в качестве метрики для оптимизации
                preds = model.predict(eval_pool)
                return root_mean_squared_error(y_val, preds)

            except Exception as e:
                logger.warning(f"Ошибка в trial {trial.number}: {str(e)}")
                return float('inf')

        try:
            self.study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(
                    seed=self.config.RANDOM_STATE),
                study_name='catboost_optimization'
            )

            self.study.optimize(
                objective,
                n_trials=n_trials,
                gc_after_trial=True,
                show_progress_bar=True
            )

            # Обучаем модель с лучшими параметрами
            best_params = self.study.best_params.copy()
            best_params.update(self.config.GPU_PARAMS)

            logger.info("Оптимизация завершена. Лучшие параметры:")
            # for param, value in best_params.items():
            #     logger.info(f"{param:>25}: {value}")

            return best_params, self.study

        except Exception as e:
            logger.error(
                f"Ошибка оптимизации гиперпараметров: {str(e)}", exc_info=True)
            raise

    def cross_validate(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       params: Dict[str, Any],
                       cat_features: List[int] = None,
                       n_splits: int = None) -> Dict[str, float]:
        """
        Проводит кросс-валидацию модели с расчетом метрик

        Args:
            X: Полный набор данных
            y: Целевая переменная
            params: Параметры модели
            cat_features: Список индексов категориальных признаков
            n_splits: Количество фолдов

        Returns:
            Словарь с усредненными метриками и их стандартными отклонениями
        """
        logger.info(
            f"Начало кросс-валидации ({n_splits or self.config.CV_FOLDS} фолдов)...")

        n_splits = n_splits or self.config.CV_FOLDS
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=self.config.RANDOM_STATE)

        metrics = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'spearman': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            try:
                model = CatBoostRegressor(**params)
                model.fit(
                    X_train, y_train,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    verbose=False
                )

                preds = model.predict(X_val)

                # Расчет метрик
                metrics['rmse'].append(
                    root_mean_squared_error(y_val, preds))
                metrics['mae'].append(mean_absolute_error(y_val, preds))
                metrics['r2'].append(r2_score(y_val, preds))
                metrics['spearman'].append(spearmanr(y_val, preds)[0])

                logger.info(
                    f"Fold {fold}: "
                    f"RMSE={metrics['rmse'][-1]:.4f}, "
                    f"MAE={metrics['mae'][-1]:.4f}, "
                    f"R2={metrics['r2'][-1]:.4f}, "
                    f"Spearman={metrics['spearman'][-1]:.4f}"
                )

            except Exception as e:
                logger.error(f"Ошибка в фолде {fold}: {str(e)}")
                continue

                # Агрегируем результаты
        result = {}
        for metric, values in metrics.items():
            if values:  # Проверяем, что есть значения
                result[f'mean_{metric}'] = np.mean(values)
                result[f'std_{metric}'] = np.std(values)
            else:
                result[f'mean_{metric}'] = np.nan
                result[f'std_{metric}'] = np.nan

        logger.info("Результаты кросс-валидации:")
        for metric in ['rmse', 'mae', 'r2', 'spearman']:
            logger.info(
                f"{metric.upper():>10}: {result[f'mean_{metric}']:.4f} ± {result[f'std_{metric}']:.4f}")

        return result

    def compute_shap_values(self,
                            model: CatBoostRegressor,
                            X: pd.DataFrame,
                            cat_features: List[int] = None,
                            sample_size: int = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Вычисляет SHAP значения для интерпретации модели

        Args:
            model: Обученная модель CatBoost
            X: Данные для анализа
            cat_features: Список индексов категориальных признаков
            sample_size: Размер подвыборки для расчета

        Returns:
            Кортеж (SHAP значения, DataFrame с данными)
        """
        logger.info("Вычисление SHAP значений...")

        try:
            sample_size = sample_size or self.config.SHAP_SAMPLES
            sample_size = min(sample_size, len(X))

            # Подготовка данных
            X_shap = X.copy()
            if cat_features:
                for col_idx in cat_features:
                    col_name = X_shap.columns[col_idx]
                    X_shap[col_name] = X_shap[col_name].astype(
                        'category').cat.codes

            # Выбор подвыборки
            sample_idx = np.random.choice(
                len(X_shap), sample_size, replace=False)
            X_sample = X_shap.iloc[sample_idx]

            # Вычисление SHAP значений
            explainer = shap.TreeExplainer(model)
            self.shap_values = explainer.shap_values(X_sample)
            self.shap_data = X_sample

            logger.info(f"SHAP значения рассчитаны для {sample_size} образцов")
            return self.shap_values, self.shap_data

        except Exception as e:
            logger.error(
                f"Ошибка расчета SHAP значений: {str(e)}", exc_info=True)
            raise

    def evaluate_model(self,
                       model: CatBoostRegressor,
                       X: pd.DataFrame,
                       y: pd.Series,
                       return_dict: bool = True) -> Union[Dict[str, float], Tuple[float, float, float, float]]:
        """
        Оценивает качество модели на тестовых данных

        Args:
            model: Обученная модель
            X: Тестовые данные
            y: Истинные значения
            return_dict: Возвращать результат как словарь или кортеж

        Returns:
            Метрики качества модели
        """
        logger.info("Оценка качества модели...")

        try:
            preds = model.predict(X)

            metrics = {
                'RMSE': root_mean_squared_error(y, preds),
                'MAE': mean_absolute_error(y, preds),
                'R2': r2_score(y, preds),
                'MedianAE': median_absolute_error(y, preds),  # Добавлено
                'MAPE': np.mean(np.abs((y - preds) / y)) * 100,
                'Spearman': spearmanr(y, preds)[0]
            }

            logger.info("Метрики качества:")
            for name, value in metrics.items():
                logger.info(f"{name:>10}: {value:.4f}")

            return metrics if return_dict else tuple(metrics.values())

        except Exception as e:
            logger.error(f"Ошибка оценки модели: {str(e)}", exc_info=True)
            raise

    def get_feature_importance(self,
                               model: CatBoostRegressor = None,
                               X_val: pd.DataFrame = None,
                               y_val: pd.Series = None,
                               cat_features: List[int] = None,
                               type: str = 'prediction') -> pd.DataFrame:
        """
        Возвращает важность признаков

        Args:
            model: Модель (если None, используется self.best_model)
            type: Тип важности ('prediction', 'loss', 'shap')

        Returns:
            DataFrame с важностью признаков
        """
        logger.info(f"Получение важности признаков (тип: {type})...")

        model = model or self.best_model
        if model is None:
            raise ValueError("Модель не обучена!")
        try:
            if type == 'shap':
                if X_val is None or y_val is None:
                    raise ValueError("Для SHAP важности нужны X_val и y_val")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val)

                importance = pd.DataFrame({
                    'feature': X_val.columns,
                    'importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('importance', ascending=False)
            else:
                importance = model.get_feature_importance(type=type)
                importance = pd.DataFrame({
                    'feature': model.feature_names_,
                    'importance': importance
                }).sort_values('importance', ascending=False)

            logger.info("Топ-5 важных признаков:")
            logger.info(importance.head(5).to_string(index=False))

            return importance

        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {str(e)}")
            raise
