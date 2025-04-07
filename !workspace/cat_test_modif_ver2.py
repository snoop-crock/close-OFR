# === БЛОК 1: Импорты библиотек ===
import pandas as pd
import numpy as np
import os
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import shap

# Машинное обучение
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Оптимизация
import optuna
import joblib

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# === БЛОК 2: Настройка путей ===
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === БЛОК 3: Функции модели ===


def objective(trial, X_train, X_val, y_train, y_val, cat_features, trial_number):
    """Функция оптимизации гиперпараметров для Optuna с сохранением ошибок."""
    params = {
        'iterations': trial.suggest_int('iterations', 5000, 15000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.3, log=True),
        'depth': trial.suggest_int('depth', 6, 12),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    }

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        verbose=0,
        early_stopping_rounds=1000,
        task_type="GPU",
        devices='0'
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    y_pred = model.predict(X_val)

    # Расчет метрик
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    spearman = spearmanr(y_val, y_pred)[0]

    # Сохранение ошибок для каждого trial
    trial_results = {
        'trial_number': trial_number,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman': spearman,
        **params
    }

    # Добавляем ошибки в историю trials
    trial.set_user_attr('errors', trial_results)

    return mse


def cross_validate_model(X, y, params, cat_features, n_splits=5):
    """Кросс-валидация модели."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(
            **params, cat_features=cat_features, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        y_pred = model.predict(X_val)

        metrics.append({
            'fold': fold + 1,
            'mse': mean_squared_error(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred),
            'spearman': spearmanr(y_val, y_pred)[0]
        })

    return pd.DataFrame(metrics)


def train_final_model(X_train, y_train, X_val, y_val, best_params, cat_features):
    """Обучение финальной модели с лучшими параметрами."""
    best_params.update({
        'cat_features': cat_features,
        'verbose': 500,
        'early_stopping_rounds': 1000,
        'task_type': "GPU",
        'devices': '0'
    })

    model = CatBoostRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model

# === БЛОК 4: Функции для отчетов и визуализаций ===


def save_shap_analysis(model, X, feature_names, results_dir):
    """Анализ и визуализация SHAP значений."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_summary.png'), dpi=300)
    plt.close()

    # Feature importance
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(
        results_dir, 'shap_feature_importance.png'), dpi=300)
    plt.close()

    # Возвращаем SHAP значения для дальнейшего анализа
    return shap_values


def save_plots(study, X_train, y_val, y_pred, shap_values=None):
    """Сохранение всех графиков."""
    # График сравнения предсказаний
    plt.figure(figsize=(10, 6))
    sns.regplot(x=y_val, y=y_pred, scatter_kws={'alpha': 0.3})
    plt.plot([y_val.min(), y_val.max()], [
             y_val.min(), y_val.max()], 'k--', lw=2)
    plt.title('Сравнение истинных и предсказанных значений')
    plt.savefig(os.path.join(RESULTS_DIR, 'true_vs_predicted.png'), dpi=300)
    plt.close()

    # Матрица корреляций (Pearson и Spearman)
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Матрица корреляций Пирсона')
    plt.savefig(os.path.join(RESULTS_DIR, 'pearson_correlation.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(X_train.corr(method='spearman'),
                annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Матрица корреляций Спирмана')
    plt.savefig(os.path.join(RESULTS_DIR, 'spearman_correlation.png'), dpi=300)
    plt.close()

    # Визуализации Optuna
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(RESULTS_DIR, "optimization_history.html"))

        fig = plot_param_importances(study)
        fig.write_html(os.path.join(RESULTS_DIR, "param_importances.html"))
    except ImportError:
        print("Plotly не установлен, пропускаем визуализации Optuna")


def generate_reports(study, model, X_train, X_val, y_val, y_pred, feature_analysis, cv_results=None):
    """Генерация всех отчетов."""
    # Сохранение модели и исследования
    model.save_model(os.path.join(RESULTS_DIR, "catboost_model.cbm"))
    joblib.dump(study, os.path.join(RESULTS_DIR, 'optuna_study.pkl'))

    with pd.ExcelWriter(os.path.join(RESULTS_DIR, 'full_report.xlsx')) as writer:
        # Лист с важностью признаков
        feature_analysis['Рекомендация'] = np.where(
            feature_analysis['Пермутационная важность'] < feature_analysis['Пермутационная важность'].quantile(
                0.25),
            'Удалить',
            'Сохранить'
        )
        feature_analysis.to_excel(
            writer, sheet_name='Feature_Importance', index=False)

        # Лист с ошибками финальной модели
        error_matrix = pd.DataFrame({
            'Истинное значение': y_val.values,
            'Предсказанное значение': y_pred,
            'Абсолютная ошибка': np.abs(y_val - y_pred),
            'Процент ошибки': np.abs((y_val - y_pred) / y_val) * 100,
            'Spearman correlation': spearmanr(y_val, y_pred)[0]
        })
        error_matrix.to_excel(writer, sheet_name='Error_Analysis', index=False)

        # Лист с гиперпараметрами
        pd.DataFrame([study.best_params]).to_excel(
            writer, sheet_name='Hyperparameters', index=False)

        # Лист с матрицами корреляций
        X_train.corr().to_excel(writer, sheet_name='Pearson_Correlation')
        X_train.corr(method='spearman').to_excel(
            writer, sheet_name='Spearman_Correlation')

        # Лист с логами оптимизации и ошибками
        optimization_log = study.trials_dataframe()

        # Извлекаем ошибки из каждого trial
        errors_list = []
        for trial in study.trials:
            if trial.user_attrs.get('errors'):
                errors_list.append(trial.user_attrs['errors'])

        errors_df = pd.DataFrame(errors_list)
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name='Trial_Errors', index=False)

        optimization_log.to_excel(
            writer, sheet_name='Optimization_Log', index=False)

        # Лист с результатами кросс-валидации
        if cv_results is not None:
            cv_results.to_excel(writer, sheet_name='CV_Results', index=False)
            cv_summary = cv_results.describe().loc[['mean', 'std']]
            cv_summary.to_excel(writer, sheet_name='CV_Summary')

# === БЛОК 5: Основной пайплайн ===


def process_and_train_model(file_path, test_size=0.1, n_trials=100, n_cv_splits=5):
    """Полный пайплайн обработки данных и обучения модели."""
    # Загрузка и подготовка данных
    data = pd.read_excel(file_path)
    if 'OFR' not in data.columns:
        raise KeyError("Файл не содержит столбца 'Optimized Flow Rate'.")

    X = data.drop(columns=['OFR']).dropna()
    y = data.loc[X.index, 'OFR']
    feature_names = X.columns.tolist()

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)
    cat_features = [i for i, col in enumerate(
        X.columns) if X[col].dtype == 'object']

    # Оптимизация гиперпараметров
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    # Добавляем номер trial для отслеживания
    trial_counter = 0

    def wrapped_objective(trial):
        nonlocal trial_counter
        trial_counter += 1
        return objective(trial, X_train, X_val, y_train, y_val, cat_features, trial_counter)

    study.optimize(
        wrapped_objective,
        n_trials=n_trials,
        gc_after_trial=True
    )

    # Обучение финальной модели
    model = train_final_model(X_train, y_train, X_val,
                              y_val, study.best_params.copy(), cat_features)

    # Оценка модели
    y_pred = model.predict(X_val)
    metrics = {
        'MSE': mean_squared_error(y_val, y_pred),
        'MAE': mean_absolute_error(y_val, y_pred),
        'R2': r2_score(y_val, y_pred),
        'Spearman': spearmanr(y_val, y_pred)[0]
    }

    print("\nФинальные метрики модели:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Кросс-валидация
    print("\nПроведение кросс-валидации...")
    cv_results = cross_validate_model(
        X_train, y_train, study.best_params, cat_features, n_splits=n_cv_splits)
    print("\nРезультаты кросс-валидации:")
    print(cv_results.describe().loc[['mean', 'std']])

    # SHAP анализ
    print("\nПроведение SHAP анализа...")
    shap_values = save_shap_analysis(model, X_val, feature_names, RESULTS_DIR)

    # Анализ признаков
    perm_importance = permutation_importance(model, X_val, y_val, n_repeats=10)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    feature_analysis = pd.DataFrame({
        'Feature': X.columns[sorted_idx],
        'Permutation Importance': perm_importance.importances_mean[sorted_idx],
        'Model Importance': model.get_feature_importance()[sorted_idx],
        'SHAP Importance': np.abs(shap_values.values).mean(0)[sorted_idx],
        'Type': ['Categorical' if i in cat_features else 'Numerical' for i in sorted_idx]
    })

    # Генерация отчетов
    generate_reports(study, model, X_train, X_val, y_val,
                     y_pred, feature_analysis, cv_results)
    save_plots(study, X_train, y_val, y_pred, shap_values)

    return model, feature_analysis, cv_results


# === БЛОК 6: Запуск приложения ===
if __name__ == "__main__":
    model, feature_analysis, cv_results = process_and_train_model(
        file_path="Normalize_avg.xlsx",
        test_size=0.1,
        n_trials=1,
        n_cv_splits=5
    )

    print("\nРезультаты сохранены в папку 'results':")
    print("- Модель: catboost_model.cbm")
    print("- Отчеты: full_report.xlsx (содержит 7 листов)")
    print("- Графики:")
    print("  * true_vs_predicted.png")
    print("  * pearson_correlation.png")
    print("  * spearman_correlation.png")
    print("  * shap_summary.png")
    print("  * shap_feature_importance.png")
    print("- Визуализации Optuna: *.html")
    print("- Исследование Optuna: optuna_study.pkl")
