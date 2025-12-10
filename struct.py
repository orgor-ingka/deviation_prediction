# This script prints a suggested repository structure for a machine learning project.
# It covers dataset building (BigQuery), model training, testing, and deployment.

def print_repo_structure():
    structure = """
.  # Project root
├── README.md             # Project overview, setup instructions
├── requirements.txt      # Python dependencies
├── .gitignore            # Files/directories to ignore in Git
├── data/
│   ├── raw/              # Raw data (e.g., BigQuery export results)
│   └── processed/        # Cleaned, preprocessed datasets
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline/
│   │   ├── __init__.py
│   │   ├── bigquery_extractor.py  # Code to query and extract data from BigQuery
│   │   └── preprocessor.py        # Code to clean and transform data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py               # Script for model training
│   │   ├── predict.py             # Script for making predictions
│   │   └── model_architecture.py  # Model definition (e.g., neural network architecture)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── test.py                # Script for model testing and evaluation metrics
│   └── utils/
│       └── helpers.py             # Common utility functions
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_model.py
├── deployment/
│   ├── Dockerfile                 # Dockerfile for containerizing the model service
│   ├── app.py                     # API endpoint for model inference (e.g., Flask/FastAPI)
│   ├── cloud_config.yaml          # Cloud-specific deployment configurations (e.g., GCP Cloud Run, AWS Lambda)
│   └── scripts/
│       └── deploy.sh              # Script to automate deployment
├── config/
│   ├── config.yaml                # General project configurations (e.g., BigQuery project ID, model parameters)
│   └── credentials.json           # Service account keys or reference to secrets manager
└── scripts/
    ├── run_training.sh            # Shell script to trigger training
    └── run_data_pipeline.sh       # Shell script to trigger data extraction/preprocessing
"""
    print(structure)

# Call the function to print the suggested structure
print_repo_structure()
