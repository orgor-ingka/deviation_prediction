#!/bin/bash

# Create main directories
mkdir -p data notebooks src tests deployment config scripts

# Create files in the root directory
touch README.md requirements.txt .gitignore

# Create subdirectories and files within 'data'
mkdir -p data/raw data/processed

# Create subdirectories and files within 'notebooks'
touch notebooks/exploratory_data_analysis.ipynb notebooks/model_prototyping.ipynb

# Create subdirectories and files within 'src'
mkdir -p src/data_pipeline src/models src/evaluation src/utils
touch src/__init__.py
touch src/data_pipeline/__init__.py src/data_pipeline/bigquery_extractor.py src/data_pipeline/preprocessor.py
touch src/models/__init__.py src/models/train.py src/models/predict.py src/models/model_architecture.py
touch src/evaluation/__init__.py src/evaluation/test.py
touch src/utils/helpers.py

# Create subdirectories and files within 'tests'
touch tests/__init__.py tests/test_data_pipeline.py tests/test_model.py

# Create subdirectories and files within 'deployment'
mkdir -p deployment/scripts
touch deployment/Dockerfile deployment/app.py deployment/cloud_config.yaml
touch deployment/scripts/deploy.sh

# Create subdirectories and files within 'config'
touch config/config.yaml config/credentials.json

# Create files within 'scripts'
touch scripts/run_training.sh scripts/run_data_pipeline.sh

echo "ML repository structure created successfully!"
