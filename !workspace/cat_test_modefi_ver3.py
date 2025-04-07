# === БЛОК 1: Импорты и конфигурация ===
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostRegressor
from optuna.visualization import plot_optimization_history, plot_param_importances

# Конфигурация


class Config:
    RESULTS_DIR = "results"
    RANDOM_STATE = 42
    TEST_SIZE = 0.1
    N_TRIALS = 2  # Увеличено для работы визуализаций Optuna
    N_CV_SPLITS = 5
    EARLY_STOPPING_ROUNDS = 1000
    FILE_PATH = "Normalize_avg.xlsx"
    TARGET_COL = "OFR"
    GPU_PARAMS = {
        'task_type': "GPU",
        'devices': '0:1'  # Использование первого GPU
    }


os.makedirs(Config.RESULTS_DIR, exist_ok=True)

# === БЛОК 2: Вспомогательные функции ===


def calculate_metrics(y_true, y_pred):
    """Вычисление всех метрик качества."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'Spearman': spearmanr(y_true, y_pred)[0]
    }


def print_metrics(metrics, title="Метрики модели"):
    """Печать метрик в удобном формате."""
    print(f"\n{title}:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

# === БЛОК 3: Функции модели ===


def get_catboost_params(trial):
    """Генерация параметров CatBoost для Optuna."""
    params = {
        'iterations': trial.suggest_int('iterations', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 3),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise'])
    }

    # Добавляем GPU параметры
    params.update(Config.GPU_PARAMS)
    return params


def objective(trial, X_train, X_val, y_train, y_val, cat_features, trial_number):
    """Функция оптимизации гиперпараметров для Optuna."""
    params = get_catboost_params(trial)

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        verbose=0,
        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    y_pred = model.predict(X_val)

    metrics = calculate_metrics(y_val, y_pred)
    trial.set_user_attr(
        'errors', {'trial_number': trial_number, **metrics, **params})

    return metrics['MSE']


def cross_validate_model(X, y, params, cat_features, n_splits=Config.N_CV_SPLITS):
    """Кросс-валидация модели."""
    # Проверка количества фолдов
    n_splits = min(n_splits, len(y) // 2)  # Минимум 2 образца в каждом фолде

    kf = KFold(n_splits=n_splits, shuffle=True,
               random_state=Config.RANDOM_STATE)
    metrics = []

    # Удаляем cat_features из params, если он там есть
    params = params.copy()
    params.pop('cat_features', None)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            cat_features=cat_features
        )

        y_pred = model.predict(X_val)
        metrics.append({'fold': fold + 1, **calculate_metrics(y_val, y_pred)})

    return pd.DataFrame(metrics)

# === БЛОК 4: Функции визуализации и отчетов ===


def save_plot(filename, figsize=(10, 6), dpi=300):
    """Декоратор для сохранения графиков."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            plt.figure(figsize=figsize)
            func(*args, **kwargs)
            plt.tight_layout()
            plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=dpi)
            plt.close()
        return wrapper
    return decorator


@save_plot('true_vs_predicted.png')
def plot_true_vs_predicted(y_true, y_pred):
    """График сравнения предсказаний."""
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'alpha': 0.3})
    plt.plot([y_true.min(), y_true.max()], [
             y_true.min(), y_true.max()], 'k--', lw=2)
    plt.title('Сравнение истинных и предсказанных значений')


@save_plot('correlation_matrix.png', figsize=(12, 10))
def plot_correlation_matrix(X, method='pearson'):
    """Матрица корреляций."""
    sns.heatmap(X.corr(method=method), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f'Матрица корреляций ({method.capitalize()})')


def save_shap_analysis(model, X, feature_names):
    """Анализ и визуализация SHAP значений."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    @save_plot('shap_summary.png', figsize=(10, 8))
    def _plot_shap_summary():
        shap.summary_plot(
            shap_values, X, feature_names=feature_names, show=False)

    @save_plot('shap_feature_importance.png')
    def _plot_shap_importance():
        shap.plots.bar(shap_values, show=False)

    _plot_shap_summary()
    _plot_shap_importance()

    return shap_values


def generate_reports(study, model, X_train, X_val, y_val, y_pred, feature_analysis, cv_results=None):
    """Генерация всех отчетов."""
    try:
        # Сохранение моделей и данных
        model.save_model(os.path.join(
            Config.RESULTS_DIR, "catboost_model.cbm"))
        joblib.dump(study, os.path.join(
            Config.RESULTS_DIR, 'optuna_study.pkl'))

        # Создаем Excel writer
        writer = pd.ExcelWriter(os.path.join(
            Config.RESULTS_DIR, 'full_report.xlsx'), engine='openpyxl')

        # Feature Importance
        if 'Permutation Importance' in feature_analysis.columns:
            feature_analysis['Recommendation'] = np.where(
                feature_analysis['Permutation Importance'] < feature_analysis['Permutation Importance'].quantile(
                    0.25),
                'Remove', 'Keep')
        feature_analysis.to_excel(
            writer, sheet_name='Feature_Importance', index=False)

        # Error Analysis
        error_df = pd.DataFrame({
            'True Value': y_val.values,
            'Predicted Value': y_pred,
            'Absolute Error': np.abs(y_val - y_pred),
            'Percentage Error': np.where(y_val != 0, np.abs((y_val - y_pred) / y_val) * 100, np.nan),
            'Spearman Correlation': spearmanr(y_val, y_pred)[0]
        })
        error_df.to_excel(writer, sheet_name='Error_Analysis', index=False)

        # Hyperparameters (только если есть trials)
        if len(study.trials) > 0:
            pd.DataFrame([study.best_params]).to_excel(
                writer, sheet_name='Hyperparameters', index=False)

        # Correlation Matrices
        X_train.corr().to_excel(writer, sheet_name='Pearson_Correlation')
        X_train.corr(method='spearman').to_excel(
            writer, sheet_name='Spearman_Correlation')

        # Optimization Log (только если есть trials)
        if len(study.trials) > 0:
            study.trials_dataframe().to_excel(
                writer, sheet_name='Optimization_Log', index=False)

            # Trial Errors (только если есть ошибки)
            errors_df = pd.DataFrame(
                [t.user_attrs['errors'] for t in study.trials if 'errors' in t.user_attrs])
            if not errors_df.empty:
                errors_df.to_excel(
                    writer, sheet_name='Trial_Errors', index=False)

        # Cross-Validation Results
        if cv_results is not None and not cv_results.empty:
            cv_results.to_excel(writer, sheet_name='CV_Results', index=False)
            cv_summary = cv_results.describe().loc[['mean', 'std']]
            cv_summary.to_excel(writer, sheet_name='CV_Summary')

        # Гарантируем, что хотя бы один лист будет видимым
        if len(writer.book.sheetnames) == 0:
            pd.DataFrame(['No data available']).to_excel(
                writer, sheet_name='Summary', index=False)

        writer.save()
    except Exception as e:
        print(f"Ошибка при генерации отчетов: {e}")
        raise

# === БЛОК 5: Основной пайплайн ===


def run_experiment():
    """Полный пайплайн эксперимента."""
    # Загрузка данных
    try:
        data = pd.read_excel(Config.FILE_PATH)
        if Config.TARGET_COL not in data.columns:
            raise KeyError(f"Файл не содержит столбца '{Config.TARGET_COL}'")

        X = data.drop(columns=[Config.TARGET_COL]).dropna()
        y = data.loc[X.index, Config.TARGET_COL]

        # Автоматическое определение категориальных признаков
        cat_features = [
            i for i, col in enumerate(X.columns)
            if X[col].dtype == 'object' or X[col].nunique() < 10
        ]
        print(f"Обнаружено {len(cat_features)} категориальных признаков")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        raise

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)

    # Оптимизация гиперпараметров
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE))

    trial_counter = 0

    def wrapped_objective(trial):
        nonlocal trial_counter
        trial_counter += 1
        return objective(trial, X_train, X_val, y_train, y_val, cat_features, trial_counter)

    try:
        study.optimize(wrapped_objective,
                       n_trials=Config.N_TRIALS, gc_after_trial=True)
    except Exception as e:
        print(f"Ошибка оптимизации гиперпараметров: {e}")
        raise

    # Обучение финальной модели
    best_params = study.best_params.copy()
    best_params.pop('cat_features', None)  # Удаляем cat_features из параметров

    model = CatBoostRegressor(
        **best_params,
        verbose=500,
        early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
        **Config.GPU_PARAMS
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
        cat_features=cat_features
    )
    y_pred = model.predict(X_val)

    # Оценка модели
    metrics = calculate_metrics(y_val, y_pred)
    print_metrics(metrics, "Финальные метрики модели")

    # Кросс-валидация
    try:
        cv_results = cross_validate_model(
            X_train, y_train, best_params, cat_features)
        print_metrics(cv_results[['MSE', 'MAE', 'R2', 'Spearman']].mean(
        ), "Средние метрики кросс-валидации")
    except Exception as e:
        print(f"Ошибка кросс-валидации: {e}")
        cv_results = None

    # SHAP анализ
    try:
        shap_values = save_shap_analysis(model, X_val, X.columns.tolist())
    except Exception as e:
        print(f"Ошибка SHAP анализа: {e}")
        shap_values = None

    # Анализ признаков
    try:
        perm_importance = permutation_importance(
            model, X_val, y_val, n_repeats=5)
        sorted_idx = perm_importance.importances_mean.argsort()[::-1]

        feature_analysis = pd.DataFrame({
            'Feature': X.columns[sorted_idx],
            'Model Importance': model.get_feature_importance()[sorted_idx],
            'Type': ['Categorical' if i in cat_features else 'Numerical' for i in sorted_idx]
        })

        if hasattr(perm_importance, 'importances_mean'):
            feature_analysis['Permutation Importance'] = perm_importance.importances_mean[sorted_idx]

        if shap_values is not None:
            feature_analysis['SHAP Importance'] = np.abs(
                shap_values.values).mean(0)[sorted_idx]
    except Exception as e:
        print(f"Ошибка анализа признаков: {e}")
        feature_analysis = pd.DataFrame()

    # Визуализации
    try:
        plot_true_vs_predicted(y_val, y_pred)
        plot_correlation_matrix(X_train, 'pearson')
        plot_correlation_matrix(X_train, 'spearman')
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

    # Визуализации Optuna
    try:
        if len(study.trials) > 1:
            fig = plot_optimization_history(study)
            fig.write_html(os.path.join(Config.RESULTS_DIR,
                           "optimization_history.html"))

            fig = plot_param_importances(study)
            fig.write_html(os.path.join(
                Config.RESULTS_DIR, "param_importances.html"))
        else:
            print("Для визуализаций Optuna требуется более одного trial")
    except Exception as e:
        print(f"Ошибка визуализации Optuna: {e}")

    # Генерация отчетов
    try:
        generate_reports(study, model, X_train, X_val, y_val,
                         y_pred, feature_analysis, cv_results)
    except Exception as e:
        print(f"Ошибка генерации отчетов: {e}")
        raise

    return model, feature_analysis, cv_results


# === БЛОК 6: Запуск приложения ===
if __name__ == "__main__":
    try:
        model, feature_analysis, cv_results = run_experiment()

        print("\nРезультаты сохранены в папку 'results':")
        print("- Модель: catboost_model.cbm")
        print("- Отчеты: full_report.xlsx")
        print("- Графики: true_vs_predicted.png, *_correlation.png, shap_*.png")
        if Config.N_TRIALS > 1:
            print("- Визуализации Optuna: *.html")
        print("- Исследование Optuna: optuna_study.pkl")
    except Exception as e:
        print(f"\nКритическая ошибка выполнения: {e}")
