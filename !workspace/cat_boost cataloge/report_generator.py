import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from config import Config
from utils.logger import setup_logger
import shap
from scipy.stats import spearmanr
import matplotlib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pandas.plotting import table

# Используем агрегативный бэкенд для matplotlib
matplotlib.use('Agg')
logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)


class ReportGenerator:
    """Класс для генерации отчетов и визуализаций модели"""

    def __init__(self):
        self.config = Config()
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Создает директорию для отчетов если она не существует"""
        os.makedirs(self.config.REPORTS_DIR, exist_ok=True)
        logger.info(f"Директория для отчетов: {self.config.REPORTS_DIR}")

    def save_all_results(self,
                         model: object,
                         metrics: Dict[str, float],
                         study: Optional[object] = None,
                         feature_importance: Optional[pd.DataFrame] = None,
                         X_train: Optional[pd.DataFrame] = None,
                         y_train: Optional[pd.Series] = None) -> None:
        """
        Сохраняет все результаты работы модели (основной метод)

        Args:
            model: Обученная модель
            metrics: Словарь метрик
            study: Объект Optuna study
            feature_importance: DataFrame с важностью признаков
            X_train: Обучающие данные
            y_train: Целевые значения
        """
        try:
            # 1. Сохраняем отчет в Excel
            self.save_model_report(model, metrics, study, feature_importance)

            # 2. Сохраняем визуализации если есть данные
            if X_train is not None and y_train is not None:
                self.save_visualizations(model, X_train, y_train)

                # 3. SHAP анализ если есть категориальные признаки
                if feature_importance is not None and 'cat_features' in model.get_params():
                    self.save_shap_plots(
                        model, X_train, model.get_params()['cat_features'])

            logger.info("Все отчеты успешно сохранены")
        except Exception as e:
            logger.error(
                f"Ошибка при сохранении результатов: {str(e)}", exc_info=True)
            raise

    def save_model_report(self,
                          model: object,
                          metrics: Dict[str, float],
                          study: Optional[object] = None,
                          feature_importance: Optional[pd.DataFrame] = None,
                          output_dir: Optional[str] = None) -> None:
        """
        Сохраняет полный отчет о модели в формате Excel

        Args:
            model: Обученная модель
            metrics: Словарь с метриками качества
            study: Объект исследования Optuna (опционально)
            feature_importance: DataFrame с важностью признаков (опционально)
            output_dir: Директория для сохранения (по умолчанию из конфига)
        """
        output_dir = output_dir or self.config.REPORTS_DIR
        report_path = Path(output_dir) / 'model_report.xlsx'

        logger.info(f"Создание отчета модели в {report_path}")

        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # Сохраняем метрики
                pd.DataFrame([metrics]).to_excel(
                    writer, sheet_name='Metrics', index=False)

                # Сохраняем параметры модели
                if hasattr(model, 'get_params'):
                    pd.DataFrame([model.get_params()]).to_excel(
                        writer, sheet_name='Model_Params')

                # Сохраняем историю оптимизации
                if study is not None:
                    study.trials_dataframe().to_excel(writer, sheet_name='Optimization_History')

                # Сохраняем важность признаков
                if feature_importance is not None:
                    feature_importance.to_excel(
                        writer, sheet_name='Feature_Importance')

                # Дополнительные листы
                self._add_summary_sheet(writer, model, metrics)

            logger.info("Отчет успешно сохранен")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {str(e)}", exc_info=True)
            raise

    def save_visualizations(self, model, X, y, cat_features=None, output_dir=None):
        output_dir = Path(output_dir or self.config.REPORTS_DIR)

        # 1. Распределение целевой переменной
        plt.figure(figsize=(10, 6))
        sns.histplot(y, kde=True, bins=30, color='skyblue')
        plt.title('Распределение целевой переменной (OFR)')
        plt.xlabel('Значение OFR')
        plt.ylabel('Частота')
        plt.savefig(output_dir / 'target_distribution.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Корреляционная матрица
        plt.figure(figsize=(12, 10))
        corr = X.corr()
        # Маска для верхнего треугольника
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1)
        plt.title('Корреляция между признаками', pad=20)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.savefig(output_dir / 'feature_correlation.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Важность признаков (альтернативный вариант)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feat_importances = pd.Series(
                model.feature_importances_, index=model.feature_names_)
            feat_importances.nlargest(15).plot(kind='barh', color='teal')
            plt.title('Топ-15 важных признаков (Feature Importance)')
            plt.xlabel('Важность')
            plt.savefig(output_dir / 'feature_importance_alt.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        # 4. График зависимости предсказаний от фактических значений
        y_pred = model.predict(X)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6, color='royalblue')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.title('Фактические vs Предсказанные значения')
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказания')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'actual_vs_predicted.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 5. Анализ категориальных признаков
        if cat_features:
            for i, col in enumerate(X.columns):
                if i in cat_features or X[col].dtype == 'object':
                    plt.figure(figsize=(10, 6))

                    # Вариант 1: Boxplot
                    sns.boxplot(x=X[col], y=y, palette='viridis')
                    plt.title(f'Влияние {col} на целевую переменную')
                    plt.xticks(rotation=45)
                    plt.savefig(
                        output_dir / f'cat_boxplot_{col}.png', bbox_inches='tight', dpi=300)
                    plt.close()

                    # Вариант 2: Barplot со средними значениями
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=X[col], y=y, estimator=np.mean,
                                ci=None, palette='magma')
                    plt.title(f'Среднее значение OFR по {col}')
                    plt.xticks(rotation=45)
                    plt.savefig(
                        output_dir / f'cat_mean_{col}.png', bbox_inches='tight', dpi=300)
                    plt.close()

        # 6. Распределение ошибок
        errors = y - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=30, color='salmon')
        plt.title('Распределение ошибок предсказания')
        plt.xlabel('Ошибка (Факт - Прогноз)')
        plt.axvline(x=0, color='black', linestyle='--')
        plt.savefig(output_dir / 'prediction_errors.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def save_shap_plots(self, model, X, cat_features=None, sample_size=100, output_dir=None):
        output_dir = Path(output_dir or self.config.REPORTS_DIR)

        # Подготовка данных
        X_shap = X.copy()
        if cat_features:
            for col_idx in cat_features:
                col_name = X_shap.columns[col_idx]
                X_shap[col_name] = X_shap[col_name].astype(
                    'category').cat.codes

        # Выбор подвыборки
        sample_size = min(sample_size, len(X))
        X_sample = X_shap.sample(
            sample_size, random_state=self.config.RANDOM_STATE)

        # Вычисление SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # 1. Summary plot (глобальная важность)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title('SHAP Summary Plot', pad=20)
        plt.savefig(output_dir / 'shap_summary.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 2. Feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', pad=20)
        plt.savefig(output_dir / 'shap_feature_importance.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # 3. Dependence plots для топ-3 признаков
        mean_shap = np.abs(shap_values).mean(0)
        top_features = np.argsort(mean_shap)[-3:][::-1]

        for feat_idx in top_features:
            feat_name = X_sample.columns[feat_idx]

            # Базовый dependence plot
            plt.figure()
            shap.dependence_plot(
                feat_idx, shap_values, X_sample,
                interaction_index=None,
                show=False
            )
            plt.title(f'SHAP Dependence: {feat_name}', pad=15)
            plt.savefig(output_dir / f'shap_dependence_{feat_name}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

            # Интерактивный вариант (если нужно color by another feature)
            plt.figure()
            shap.dependence_plot(
                feat_idx, shap_values, X_sample,
                interaction_index='auto',
                show=False
            )
            plt.title(f'SHAP Interaction: {feat_name}', pad=15)
            plt.savefig(output_dir / f'shap_interaction_{feat_name}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        # 4. Force plot для нескольких примеров
        for i in range(min(3, sample_size)):  # Первые 3 примера
            plt.figure()
            shap.force_plot(
                explainer.expected_value,
                shap_values[i, :],
                X_sample.iloc[i, :],
                show=False,
                matplotlib=True
            )
            plt.title(f'SHAP Force Plot (Пример {i+1})', pad=10)
            plt.savefig(output_dir / f'shap_force_plot_{i}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

    def save_spearman_plot(self,
                           y_true: Union[pd.Series, np.ndarray],
                           y_pred: Union[pd.Series, np.ndarray],
                           output_dir: Optional[str] = None) -> float:
        """
        Сохраняет график корреляции Спирмена

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            output_dir: Директория для сохранения

        Returns:
            Коэффициент корреляции Спирмена
        """
        output_dir = output_dir or self.config.REPORTS_DIR
        spear, _ = spearmanr(y_true, y_pred)

        try:
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")

            # Основной график
            ax = sns.regplot(
                x=y_true,
                y=y_pred,
                scatter_kws={'alpha': 0.5, 'color': 'blue'},
                line_kws={'color': 'red', 'linestyle': '--'}
            )

            # Добавляем диагональ
            lims = [
                min(min(y_true), min(y_pred)),
                max(max(y_true), max(y_pred))
            ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

            # Оформление
            plt.title(f'Spearman Correlation: {spear:.3f}', fontsize=14)
            plt.xlabel('True Values', fontsize=12)
            plt.ylabel('Predictions', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                Path(output_dir) / 'spearman_correlation.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

            logger.info(
                f"График корреляции Спирмена сохранен (ρ = {spear:.3f})")
            return spear

        except Exception as e:
            logger.error(f"Ошибка создания графика корреляции: {str(e)}")
            raise

    def _save_true_vs_predicted(self,
                                y_true: Union[pd.Series, np.ndarray],
                                y_pred: Union[pd.Series, np.ndarray],
                                output_dir: str) -> None:
        """Сохраняет график фактических vs предсказанных значений"""
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.title('True vs Predicted Values')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'true_vs_predicted.png', dpi=300)
        plt.close()

    def _save_error_distribution(self,
                                 y_true: Union[pd.Series, np.ndarray],
                                 y_pred: Union[pd.Series, np.ndarray],
                                 output_dir: str) -> None:
        """Сохраняет график распределения ошибок"""
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=30)
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Prediction Error')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'error_distribution.png', dpi=300)
        plt.close()

    def _save_feature_importance(self,
                                 model: object,
                                 output_dir: str,
                                 top_n: int = 20) -> None:
        """Сохраняет график важности признаков"""
        if not hasattr(model, 'feature_importances_'):
            return

        importance = pd.DataFrame({
            'feature': model.feature_names_,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'feature_importance.png', dpi=300)
        plt.close()

    def _add_summary_sheet(self,
                           writer: pd.ExcelWriter,
                           model: object,
                           metrics: Dict[str, float]) -> None:
        """Добавляет сводный лист в Excel отчет"""
        summary_data = {
            'Model Type': [type(model).__name__],
            'Training Date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
            'Best Metric': [f"{metrics.get('RMSE', 'N/A')}"],
            'Features Count': [len(model.feature_names_) if hasattr(model, 'feature_names_') else 'N/A']
        }

        pd.DataFrame(summary_data).to_excel(
            writer,
            sheet_name='Summary',
            index=False
        )
