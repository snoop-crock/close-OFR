from config import Config
from model_training import ModelTrainer
from report_generator import ReportGenerator
from utils.data_processor import DataProcessor
from utils.logger import setup_logger


# Инициализация конфигурации
config = Config()
config.setup_dirs()

logger = setup_logger(__name__, log_dir=Config.LOGS_DIR)


def main():
    try:
        # Обработка данных
        processor = DataProcessor(Config)
        X, y, cat_features = processor.load_and_preprocess_data()
        X_train, X_val, y_train, y_val = processor.split_data(X, y)

        logger.info("Данные успешно загружены и обработаны")
        logger.info(f"Размер обучающей выборки: {len(X_train)}")
        logger.info(f"Размер валидационной выборки: {len(X_val)}")
        logger.info(f"Категориальные признаки: {cat_features}")

        # инициализация класса ModelTrainer
        trainer = ModelTrainer(config)
        # Подбор гиперпараметров
        best_params, study = trainer.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, cat_features)

        # Обучение с кросс-валидацией
        cv_results = trainer.cross_validate(X, y, best_params, cat_features)
        logger.info(
            f"CV Results - RMSE: {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        logger.info(
            f"Spearman: {cv_results['mean_spearman']:.4f} ± {cv_results['std_spearman']:.4f}")

        # Обучение финальной модели
        final_model = trainer.train_final_model(
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features
        )

        # Генерация отчетов
        reporter = ReportGenerator()
        metrics = trainer.evaluate_model(final_model, X_val, y_val)

        feature_importance = trainer.get_feature_importance(
            model=final_model,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features,
            type='shap'  # или 'prediction'/'loss'
        )

        reporter.save_all_results(
            final_model,
            metrics,
            study,
            feature_importance,
            X_train,
            y_train
        )
        logger.info("Процесс завершен успешно!")

    except Exception as e:
        logger.critical(f"Ошибка в основном процессе: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
