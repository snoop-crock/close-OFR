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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from optuna.visualization import plot_optimization_history
from io import BytesIO
import base64

# Используем агрегативный бэкенд для matplotlib
matplotlib.use('Agg')
logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)


class ReportGenerator:
    """Класс для генерации отчетов и визуализаций модели"""

    def __init__(self):
        self.config = Config()
        self._ensure_output_dir()
        self.report_data = {
            'metrics': {},
            'plots': {},
            'feature_importance': None,
            'learning_curves': None,
            'shap_values': None
        }

    def _ensure_output_dir(self) -> None:
        """Создает директорию для отчетов если она не существует"""
        os.makedirs(self.config.REPORTS_DIR, exist_ok=True)
        logger.info(f"Директория для отчетов: {self.config.REPORTS_DIR}")

# В файле report_generator.py внесем изменения в метод save_all_results:

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
            study: Объект Optuna study (опционально)
            feature_importance: DataFrame с важностью признаков (опционально)
            X_train: Обучающие данные (опционально)
            y_train: Целевые значения (опционально)
        """
        try:
            # 1. Сохраняем базовую информацию
            self.report_data['metrics'] = metrics
            self.report_data['feature_importance'] = feature_importance

            # 2. Сохраняем отчет в Excel
            self.save_model_report(model, metrics, study, feature_importance)

            # 3. Проверка данных перед построением графиков
            if X_train is not None and y_train is not None:
                # Проверяем размерности
                if len(X_train) != len(y_train):
                    raise ValueError(
                        "X_train и y_train должны иметь одинаковую длину")

                # Получаем предсказания
                y_pred = model.predict(X_train)

                # Преобразуем и проверяем данные
                try:
                    y_train_clean = np.array(y_train, dtype=np.float64)
                    y_pred_clean = np.array(y_pred, dtype=np.float64)

                    # Обработка NaN/Inf
                    y_train_clean = np.nan_to_num(
                        y_train_clean, nan=np.nanmedian(y_train_clean))
                    y_pred_clean = np.nan_to_num(
                        y_pred_clean, nan=np.nanmedian(y_pred_clean))

                    # 4. Сохраняем основные визуализации
                    self._save_true_vs_predicted(y_train_clean, y_pred_clean)
                    self._save_error_distribution(y_train_clean, y_pred_clean)
                    self._save_spearman_plot(y_train_clean, y_pred_clean)

                    # 5. Дополнительные графики
                    if hasattr(model, 'feature_importances_'):
                        self._save_feature_importance(
                            model, self.config.REPORTS_DIR)

                    # 6. SHAP анализ (если данные не слишком большие)
                    if feature_importance is not None and len(X_train) < 10000:
                        try:
                            self.save_shap_plots(model, X_train)
                        except Exception as e:
                            logger.warning(f"Ошибка SHAP анализа: {str(e)}")

                    # 7. Визуализации Optuna
                    if study is not None:
                        try:
                            # Сохраняем историю оптимизации
                            fig = plot_optimization_history(study)
                            optuna_history_path = Path(
                                self.config.REPORTS_DIR) / 'optimization_history.html'
                            fig.write_html(optuna_history_path)
                            logger.info(
                                f"График истории оптимизации сохранен в {optuna_history_path}")

                            # Сохраняем график сходимости MSE
                            self._save_mse_convergence(study)
                        except Exception as e:
                            logger.warning(
                                f"Ошибка сохранения визуализаций Optuna: {str(e)}")

                    # 8. Partial Dependence Plots
                    self._save_partial_dependence_plots(
                        model, X_train, self.config.REPORTS_DIR)

                except Exception as e:
                    logger.error(
                        f"Ошибка обработки данных для визуализаций: {str(e)}")
                    raise

            # 9. Генерация HTML отчета
            self.generate_html_report()

            logger.info("Все отчеты успешно сохранены")

        except Exception as e:
            logger.error(
                f"Критическая ошибка при сохранении результатов: {str(e)}", exc_info=True)
            raise

    def _save_learning_curves(self, model: object, output_dir: Optional[str] = None) -> None:
        """Сохраняет графики обучения и валидации"""
        try:
            if not hasattr(model, 'evals_result_'):
                logger.warning("Модель не имеет атрибута evals_result_")
                return

            evals_result = model.evals_result_
            if not evals_result:
                logger.warning("evals_result пуст")
                return

            plt.figure(figsize=(12, 6))

            # Убедимся, что evals_result - это словарь с нужными данными
            if isinstance(evals_result, dict):
                for metric_name, metric_values in evals_result.items():
                    if isinstance(metric_values, (list, np.ndarray)):
                        plt.plot(metric_values, label=metric_name)
                    elif isinstance(metric_values, dict):
                        # Обработка случая, когда metric_values - это словарь
                        if 'learn' in metric_name or 'validation' in metric_name:
                            plt.plot(metric_values['RMSE'], label=metric_name)

            plt.title('Learning Curves')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True)

            plot_path = Path(
                output_dir or self.config.REPORTS_DIR) / 'learning_curves.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            self.report_data['plots']['learning_curves'] = self._plot_to_base64(
                plot_path)

        except Exception as e:
            logger.warning(f"Не удалось сохранить learning curves: {str(e)}")

    def _save_residual_plot(self, y_true, y_pred) -> None:
        """Сохраняет график остатков"""
        try:
            residuals = y_true - y_pred

            plt.figure(figsize=(12, 6))
            sns.residplot(x=y_pred, y=residuals, lowess=True,
                          line_kws={'color': 'red', 'lw': 2})
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.axhline(y=0, color='black', linestyle='--')
            plt.grid(True)

            plot_path = Path(self.config.REPORTS_DIR) / 'residual_plot.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Добавляем в данные для HTML
            self.report_data['plots']['residual_plot'] = self._plot_to_base64(
                plot_path)

        except Exception as e:
            logger.warning(f"Не удалось сохранить residual plot: {str(e)}")

    def _save_partial_dependence_plots(self, model, X, n_features=3, output_dir=None) -> None:
        """Сохраняет Partial Dependence Plots для топ-N признаков"""
        try:
            output_dir = Path(output_dir or self.config.REPORTS_DIR)
            output_dir.mkdir(exist_ok=True, parents=True)

            # Выбираем топ-N важных признаков
            if hasattr(model, 'feature_importances_'):
                top_features = np.argsort(
                    model.feature_importances_)[-n_features:][::-1]
            else:
                top_features = range(min(n_features, X.shape[1]))

            for i in top_features:
                feat_name = X.columns[i]
                try:
                    plt.figure(figsize=(10, 6))
                    shap.partial_dependence_plot(
                        feat_name,
                        model.predict,
                        X,
                        model_expected_value=True,
                        feature_expected_value=True,
                        ice=False
                    )
                    plt.title(f'Partial Dependence Plot: {feat_name}')
                    plt.tight_layout()

                    plot_path = output_dir / f'pdp_{feat_name}.png'
                    plt.savefig(str(plot_path), bbox_inches='tight',
                                dpi=300)  # Преобразуем Path в строку
                    plt.close()

                    self.report_data['plots'][f'pdp_{feat_name}'] = self._plot_to_base64(
                        plot_path)
                except Exception as e:
                    logger.warning(
                        f"Не удалось сохранить PDP plot для {feat_name}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Не удалось сохранить PDP plots: {str(e)}")

    def _save_shap_beeswarm(self, model, X, sample_size=100) -> None:
        """Сохраняет SHAP beeswarm plot"""
        try:
            sample_size = min(sample_size, len(X))
            X_sample = X.sample(
                sample_size, random_state=self.config.RANDOM_STATE)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values, X_sample,
                              plot_type="violin", show=False)
            plt.title('SHAP Beeswarm Plot')
            plt.tight_layout()

            plot_path = Path(self.config.REPORTS_DIR) / 'shap_beeswarm.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            # Добавляем в данные для HTML
            self.report_data['plots']['shap_beeswarm'] = self._plot_to_base64(
                plot_path)
            self.report_data['shap_values'] = shap_values

        except Exception as e:
            logger.warning(
                f"Не удалось сохранить SHAP beeswarm plot: {str(e)}")

    def generate_html_report(self) -> None:
        """Генерирует интерактивный HTML отчет со всеми графиками"""
        try:
            html_path = Path(self.config.REPORTS_DIR) / 'model_report.html'

            # Конфигурация всех возможных графиков с пояснениями
            plots_config = [
                {'key': 'actual_vs_predicted_main',
                    'title': 'Actual vs Predicted (Main)', 'row': 1, 'col': 1},
                {'key': 'true_vs_predicted_alternative',
                    'title': 'True vs Predicted (Alternative)', 'row': 1, 'col': 2},
                {'key': 'feature_importance_main',
                    'title': 'Feature Importance (Main)', 'row': 2, 'col': 1},
                {'key': 'feature_importance_alternative',
                    'title': 'Feature Importance (Alternative)', 'row': 2, 'col': 2},
                {'key': 'shap_summary', 'title': 'SHAP Summary', 'row': 3, 'col': 1},
                {'key': 'shap_beeswarm', 'title': 'SHAP Beeswarm', 'row': 3, 'col': 2},
                {'key': 'residual_plot', 'title': 'Residual Plot', 'row': 4, 'col': 1},
                {'key': 'learning_curves',
                    'title': 'Learning Curves', 'row': 4, 'col': 2}
            ]

            # Фильтруем только доступные графики
            available_plots = [
                p for p in plots_config if p['key'] in self.report_data['plots']]

            if not available_plots:
                logger.warning("Нет доступных графиков для HTML отчета")
                return

            # Создаем subplots
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=[p['title'] for p in available_plots],
                specs=[[{"type": "xy"} for _ in range(2)] for _ in range(4)]
            )

            # Добавляем графики
            for plot in available_plots:
                img = self.report_data['plots'][plot['key']]
                fig.add_layout_image(
                    row=plot['row'], col=plot['col'],
                    source=f"data:image/png;base64,{img}",
                    xref=f"x{plot['row']}{plot['col']}",
                    yref=f"y{plot['row']}{plot['col']}",
                    x=0, y=1, sizex=1, sizey=1,
                    sizing="stretch", layer="below"
                )

            # Настройки макета
            fig.update_layout(
                title_text="Model Analysis Report (All Visualizations)",
                height=1600, width=1200,
                showlegend=False,
                margin=dict(l=20, r=20, t=100, b=20)
            )

            fig.write_html(html_path)
            logger.info(f"Полный HTML отчет сохранен: {html_path}")

        except Exception as e:
            logger.error(
                f"Ошибка генерации HTML отчета: {str(e)}", exc_info=True)
            raise

    def _plot_to_base64(self, plot_path: Path) -> str:
        """Конвертирует изображение в base64"""
        with open(plot_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

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

        # 1. Основной график распределения целевой переменной
        plt.figure(figsize=(10, 6))
        sns.histplot(y, kde=True, bins=30, color='skyblue')
        plt.title('Основное распределение целевой переменной (OFR)')
        plt.xlabel('Значение OFR')
        plt.ylabel('Частота')
        plt.savefig(output_dir / 'target_distribution_main.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        self.report_data['plots']['target_distribution_main'] = self._plot_to_base64(
            output_dir / 'target_distribution_main.png')

        # 2. Основная корреляционная матрица
        plt.figure(figsize=(12, 10))
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', cbar_kws={'shrink': 0.8})
        plt.title('Основная корреляционная матрица')
        plt.savefig(output_dir / 'correlation_matrix_main.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        self.report_data['plots']['correlation_matrix_main'] = self._plot_to_base64(
            output_dir / 'correlation_matrix_main.png')

        # 3. Основной график важности признаков (встроенный)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            feat_importances = pd.Series(
                model.feature_importances_, index=model.feature_names_)
            feat_importances.nlargest(15).plot(kind='barh', color='teal')
            plt.title('Основная важность признаков (встроенная)')
            plt.xlabel('Важность признака')
            plt.ylabel('Признак')
            plt.savefig(output_dir / 'feature_importance_main.png',
                        bbox_inches='tight', dpi=300)
            plt.close()
            self.report_data['plots']['feature_importance_main'] = self._plot_to_base64(
                output_dir / 'feature_importance_main.png')

        # 4. Основной график Actual vs Predicted
        y_pred = model.predict(X)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=y, y=y_pred, alpha=0.6)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.title('Основное сравнение: фактические vs предсказанные')
        plt.xlabel('Фактические значения')
        plt.ylabel('Предсказанные значения')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'actual_vs_predicted_main.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        self.report_data['plots']['actual_vs_predicted_main'] = self._plot_to_base64(
            output_dir / 'actual_vs_predicted_main.png')

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
        self.report_data['plots']['prediction_errors'] = self._plot_to_base64(
            output_dir / 'prediction_errors.png')

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
        self.report_data['plots']['shap_summary'] = self._plot_to_base64(
            output_dir / 'shap_summary.png')

        # 2. Feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance', pad=20)
        plt.savefig(output_dir / 'shap_feature_importance.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        self.report_data['plots']['shap_feature_importance'] = self._plot_to_base64(
            output_dir / 'shap_feature_importance.png')

        # 3. Dependence plots для топ-3 признаков
        mean_shap = np.abs(shap_values).mean(0)
        top_features = np.argsort(mean_shap)[-3:][::-1]

        for feat_idx in top_features:
            feat_name = X_sample.columns[feat_idx]

            # Базовый dependence plot
            plt.figure()
            shap.dependence_plot(feat_idx, shap_values,
                                 X_sample, interaction_index=None, show=False)
            plt.title(f'SHAP Dependence: {feat_name}', pad=15)
            plt.savefig(
                output_dir / f'shap_dependence_{feat_name}.png', bbox_inches='tight', dpi=300)
            plt.close()
            self.report_data['plots'][f'shap_dependence_{feat_name}'] = self._plot_to_base64(
                output_dir / f'shap_dependence_{feat_name}.png')

            # Интерактивный вариант
            plt.figure()
            shap.dependence_plot(feat_idx, shap_values,
                                 X_sample, interaction_index='auto', show=False)
            plt.title(f'SHAP Interaction: {feat_name}', pad=15)
            plt.savefig(
                output_dir / f'shap_interaction_{feat_name}.png', bbox_inches='tight', dpi=300)
            plt.close()
            self.report_data['plots'][f'shap_interaction_{feat_name}'] = self._plot_to_base64(
                output_dir / f'shap_interaction_{feat_name}.png')

        # 4. Force plot для нескольких примеров
        for i in range(min(3, sample_size)):
            plt.figure()
            shap.force_plot(explainer.expected_value,
                            shap_values[i, :], X_sample.iloc[i, :], show=False, matplotlib=True)
            plt.title(f'SHAP Force Plot (Пример {i+1})', pad=10)
            plt.savefig(
                output_dir / f'shap_force_plot_{i}.png', bbox_inches='tight', dpi=300)
            plt.close()
            self.report_data['plots'][f'shap_force_plot_{i}'] = self._plot_to_base64(
                output_dir / f'shap_force_plot_{i}.png')

    def _save_spearman_plot(self, y_true, y_pred):
        """Сохраняет график корреляции Спирмена"""
        try:
            spear, _ = spearmanr(y_true, y_pred)

            plt.figure(figsize=(10, 8))
            sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.5})
            plt.plot([min(y_true), max(y_true)], [
                     min(y_true), max(y_true)], 'r--')
            plt.title(f'Spearman Correlation: {spear:.3f}')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.grid(True, alpha=0.3)

            plot_path = Path(self.config.REPORTS_DIR) / \
                'spearman_correlation.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            self.report_data['plots']['spearman_correlation'] = self._plot_to_base64(
                plot_path)
            logger.info(
                f"График корреляции Спирмена сохранен (ρ = {spear:.3f})")

        except Exception as e:
            logger.error(
                f"Ошибка сохранения графика корреляции Спирмена: {str(e)}")

    def _save_true_vs_predicted(self, y_true, y_pred, output_dir=None):
        output_dir = Path(output_dir or self.config.REPORTS_DIR)

        try:
            # Убедимся, что данные являются numpy массивами
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            # Проверим и преобразуем типы данных
            if y_true.dtype == object:
                try:
                    y_true = y_true.astype(float)
                except ValueError as e:
                    logger.error(
                        f"Не удалось преобразовать y_true в float: {str(e)}")
                    return

            if y_pred.dtype == object:
                try:
                    y_pred = y_pred.astype(float)
                except ValueError as e:
                    logger.error(
                        f"Не удалось преобразовать y_pred в float: {str(e)}")
                    return

            # Проверим на NaN/Inf
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                logger.warning(
                    "Обнаружены NaN или Inf в предсказаниях, заменяем медианой")
                y_pred = np.where(np.isfinite(y_pred),
                                  y_pred, np.nanmedian(y_pred))

            if np.isnan(y_true).any() or np.isinf(y_true).any():
                logger.warning(
                    "Обнаружены NaN или Inf в истинных значениях, заменяем медианой")
                y_true = np.where(np.isfinite(y_true),
                                  y_true, np.nanmedian(y_true))

            # Создаем фигуру
            plt.figure(figsize=(14, 10))

            # Основной scatter plot
            plt.subplot(2, 2, 3)
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([min(y_true), max(y_true)], [
                     min(y_true), max(y_true)], 'r--')
            plt.title('Фактические vs Предсказанные значения')
            plt.xlabel('Фактические значения')
            plt.ylabel('Предсказанные значения')

            # Гистограмма ошибок
            plt.subplot(2, 2, 1)
            errors = y_true - y_pred
            plt.hist(errors, bins=30, density=True)
            plt.title('Распределение ошибок')

            # Гистограмма предсказаний
            plt.subplot(2, 2, 4)
            plt.hist(y_pred, bins=30, orientation='horizontal', density=True)
            plt.title('Распределение предсказаний')

            plt.tight_layout()
            plot_path = output_dir / 'actual_vs_predicted_alternative.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            self.report_data['plots']['actual_vs_predicted_alternative'] = self._plot_to_base64(
                plot_path)

        except Exception as e:
            logger.error(
                f"Критическая ошибка в _save_true_vs_predicted: {str(e)}", exc_info=True)
            raise

    def _save_error_distribution(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray) -> None:
        """
        Сохраняет график распределения ошибок

        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            output_dir: Директория для сохранения
        """
        try:
            errors = y_true - y_pred
            plt.figure(figsize=(10, 6))
            sns.histplot(errors, bins=30, kde=True)
            plt.title('Распределение ошибок предсказания')
            plt.xlabel('Ошибка (Факт - Прогноз)')

            plot_path = Path(self.config.REPORTS_DIR) / \
                'error_distribution.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            self.report_data['plots']['error_distribution'] = self._plot_to_base64(
                plot_path)

        except Exception as e:
            logger.error(f"Ошибка сохранения распределения ошибок: {str(e)}")
            raise

    def _save_feature_importance(self, model, output_dir=None):
        """Альтернативный график важности признаков с анализом"""
        output_dir = Path(output_dir or self.config.REPORTS_DIR)

        if not hasattr(model, 'feature_importances_'):
            return

        importance = pd.DataFrame({
            'feature': model.feature_names_,
            'importance': model.feature_importances_,
            'importance_pct': model.feature_importances_ / model.feature_importances_.sum() * 100
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        bars = sns.barplot(x='importance_pct', y='feature',
                           data=importance.head(20))

        # Добавляем значения на график
        for i, (_, row) in enumerate(importance.head(20).iterrows()):
            bars.text(row['importance_pct'] + 0.5, i, f"{row['importance_pct']:.1f}%",
                      color='black', ha="left", va="center")

        plt.title('Альтернативная важность признаков (с процентами)')
        plt.xlabel('Важность (%)')
        plt.savefig(output_dir / 'feature_importance_alternative.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        self.report_data['plots']['feature_importance_alternative'] = self._plot_to_base64(
            output_dir / 'feature_importance_alternative.png')

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

    def _save_mse_convergence(self, study, output_dir=None):
        """Сохраняет график сходимости MSE в процессе оптимизации"""
        try:
            if study is None:
                return

            trials_df = study.trials_dataframe()
            if trials_df.empty:
                return

            plt.figure(figsize=(12, 6))
            plt.plot(trials_df['number'],
                     trials_df['value'], 'b-', label='MSE')
            plt.axhline(y=study.best_value, color='r', linestyle='--',
                        label=f'Лучшее MSE: {study.best_value:.4f}')
            plt.title('Сходимость MSE в процессе оптимизации')
            plt.xlabel('Номер trial')
            plt.ylabel('MSE')
            plt.legend()
            plt.grid(True)

            plot_path = Path(
                output_dir or self.config.REPORTS_DIR) / 'mse_convergence.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()

            self.report_data['plots']['mse_convergence'] = self._plot_to_base64(
                plot_path)
        except Exception as e:
            logger.warning(
                f"Не удалось сохранить график сходимости MSE: {str(e)}")
