# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Leonard Umoru',
#     license='',
# )

import logging
import traceback
from src.data.make_dataset import load_data
from src.visualization.visualize import plot_violin
from src.features.build_features import feature_eng
from src.models.train_model import train_lrmodel
from src.models.predict_model import eval_model

# Set up logging
logging.basicConfig(
    filename="real_estate_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

if __name__ == "__main__":
    try:
        logging.info("Real Estate pipeline started.")

        # Load and preprocess the data
        data_path = "data/raw/real_estate.csv"
        try:
            df = load_data(data_path)
            logging.info(f"Data loaded successfully from {data_path}")
        except Exception as e:
            logging.error("Failed to load data.")
            logging.error(traceback.format_exc())
            raise

        # Create dummy variables and separate features and target
        try:
            x, y = feature_eng(df)
            logging.info("Feature engineering completed successfully.")
        except Exception as e:
            logging.error("Error during feature engineering.")
            logging.error(traceback.format_exc())
            raise

        # Train the linear regression model
        try:
            lrmodel, x_train, x_test, y_train, y_test = train_lrmodel(x, y)
            logging.info("Linear regression model trained successfully.")
        except Exception as e:
            logging.error("Error during model training.")
            logging.error(traceback.format_exc())
            raise

        # Visualization
        try:
            plot_violin(df)
            logging.info("Violin plot created successfully.")
        except Exception as e:
            logging.warning("Failed to generate violin plot.")
            logging.warning(traceback.format_exc())

        # Model evaluation
        try:
            train_mae, test_mae = eval_model(lrmodel, x_train, x_test, y_train, y_test)
            logging.info(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
            print(f"Train error is, {train_mae}")
            print(f"Test error is, {test_mae}")
        except Exception as e:
            logging.error("Error during model evaluation.")
            logging.error(traceback.format_exc())
            raise

        logging.info("Real Estate pipeline completed successfully.")

    except Exception as e:
        logging.critical("Pipeline execution failed.")
        logging.critical(traceback.format_exc())

