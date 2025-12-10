from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import optuna

def define_model_architecture(model_type: str, **kwargs):
    """
    Defines and returns the architecture or an instance of a specified machine learning model,
    along with a factory function to create an Optuna objective for hyperparameter tuning.

    Args:
        model_type (str): The type of model to define. Options: 'random_forest',
                          'balanced_random_forest', 'xgboost', 'neural_network'.
        **kwargs: Arbitrary keyword arguments to pass to the model constructor or objective factory.

    Returns:
        dict: A dictionary containing:
            - 'model_definition': An instance of the specified model or a representation of its architecture.
            - 'objective_factory': A function that takes (X_train, y_train, X_val, y_val) and returns
                                   an Optuna objective function for hyperparameter tuning.
    """
    print(f"Defining model architecture and objective factory for: {model_type}")

    model_definition = None
    objective_factory = None

    if model_type == 'random_forest':
        print("Using Random Forest Classifier.")
        # Model definition
        model_definition = RandomForestClassifier(**kwargs)

        # Objective factory for Optuna
        def rf_objective_factory(X_train, y_train, X_val, y_val):
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_depth = trial.suggest_int('max_depth', 2, 32)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                                class_weight='balanced', random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                accuracy = model.score(X_val, y_val)
                return accuracy
            
                print(f"Optuna trial for RandomForest: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, min_samples_split={min_samples_split}")
                return 0.5 # Dummy accuracy
            
            return objective
        objective_factory = rf_objective_factory

    elif model_type == 'balanced_random_forest':
        print("Using Balanced Random Forest Classifier.")
        model_definition = BalancedRandomForestClassifier(**kwargs)

        def brf_objective_factory(X_train, y_train, X_val, y_val):
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_depth = trial.suggest_int('max_depth', 2, 32)
                sampling_strategy = trial.suggest_categorical('sampling_strategy', ['auto', 'all', 'not majority'])
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
                
                model = BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                       sampling_strategy=sampling_strategy, min_samples_split=min_samples_split, 
                                                       min_samples_leaf=min_samples_leaf, max_features=max_features, class_weight='balanced',
                                                       random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                accuracy = model.score(X_val, y_val)
                return accuracy
                # print(f"Optuna trial for BalancedRandomForest: n_estimators={n_estimators}, max_depth={max_depth}, sampling_strategy={sampling_strategy}")
                # return 0.6 # Dummy accuracy
            return objective
        objective_factory = brf_objective_factory

    elif model_type == 'xgboost':
        print("Using XGBoost Classifier.")
        # model_definition = xgb.XGBClassifier(**kwargs)
        model_definition = f"XGBClassifier with params: {kwargs}"

        def xgb_objective_factory(X_train, y_train, X_val, y_val):
            def objective(trial):
                n_estimators = trial.suggest_int('n_estimators', 100, 1000)
                max_depth = trial.suggest_int('max_depth', 2, 10)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
                
                # Calculate scale_pos_weight based on class distribution
                # Ensure y_train is a pandas Series or numpy array for .sum() to work as expected
                positive_class_count = (y_train == 1).sum()
                negative_class_count = (y_train == 0).sum()
                scale_pos_weight = negative_class_count / positive_class_count if positive_class_count > 0 else 1

                model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          learning_rate=learning_rate, scale_pos_weight=scale_pos_weight,
                                          use_label_encoder=False, eval_metric='logloss', random_state=42)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
                accuracy = model.score(X_val, y_val)
                return accuracy
                # print(f"Optuna trial for XGBoost: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, scale_pos_weight={scale_pos_weight}")
                # return 0.7 # Dummy accuracy
            return objective
        objective_factory = xgb_objective_factory


    else:
        print(f"Error: Unknown model type '{model_type}'.")
        return {'model_definition': None, 'objective_factory': None}

    return {'model_definition': model_definition, 'objective_factory': objective_factory}