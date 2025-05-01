# MLflow Model Trainer and Importer

This repository contains two Python scripts designed to help with model training and registration in [MLflow](https://mlflow.org/):

- `mlflow_trainer.py`: Train a machine learning model and register it with the MLflow Model Registry.
- `import_models.py`: Import an already trained model into the MLflow Model Registry.

---

## Requirements

Ensure the following Python packages are installed:

```bash
pip install pandas numpy scikit-learn tensorflow mlflow openpyxl
```

You must also have an MLflow tracking server running.

## 1. Training and Registering a Model (`mlflow_trainer.py`)

This script handles:

- Data preprocessing (including missing value imputation, scaling, and encoding)
- Neural network model creation (for classification, regression, or autoencoder-based tasks)
- Training the model with TensorFlow/Keras
- Saving artifacts (model + scaler)
- Logging and registering the model to MLflow

### Usage

```bash
python mlflow_trainer.py \
  --file_path <path_to_csv_or_excel> \
  --model_name <mlflow_model_name> \
  --task_type <classification|regression|unsupervised classification|anomaly detection> \
  --save_path <output_model_save_path> \
  [--label <target_column_name>] \
  [--description <optional_description>] \
  [--epochs <num_epochs>] \
  [--batch_size <batch_size>] \
  [--mlflow_tracking_uri <mlflow_tracking_uri>]
```

### Example use

```bash
python mlflow_trainer.py --file_path churn_train_data.xlsx --model_name CustomerChurn1 --task_type classification --save_path churn_model/model.h5 --label Churn --description "Customer churn prediction model" --epochs 25 --mlflow_tracking_uri http://localhost:5001
```

## 2. Importing a Pre-Trained Model (`import_models.py`)

This script allows you to register an already trained Keras/TensorFlow model with MLflow. It also logs related artifacts such as scalers or encoders stored as `.pkl` files.

### Usage

```bash
python import_models.py \
  --model_path <path_to_saved_model> \
  --model_name <mlflow_model_name> \
  --task_type <classification|regression|unsupervised classification|anomaly detection> \
  [--description <optional_description>] \
  [--artifacts_dir <directory_with_pkl_files>] \
  [--mlflow_tracking_uri <mlflow_tracking_uri>]
```

### Example use

```bash
python import_models.py --model_path ./churn_model/model.h5 --model_name "ChurnModel1" --task_type "classification" --artifacts_dir ./churn_model/ --description "Predicts if a customer will leave based on behavior patterns"
```
