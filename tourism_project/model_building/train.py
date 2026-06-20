# for data manipulation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.stats import randint, uniform
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/cooldude101983/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/cooldude101983/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/cooldude101983/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/cooldude101983/tourism-package-prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
]


# 2. Preprocess (no scaling needed for trees; OHE for cats)
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])


# 3. HistGB with built-in regularization (l2, early stopping, leaf limits)
pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        class_weight='balanced'))
])

# 4. Search over REGULARIZING params to shrink the gap
param_dist = {
    'clf__learning_rate':     uniform(0.02, 0.08),
    'clf__max_iter':          randint(150, 400),
    'clf__max_depth':         [3, 4],
    'clf__max_leaf_nodes':    randint(10, 24),
    'clf__min_samples_leaf':  randint(25, 70),
    'clf__l2_regularization': uniform(0.5, 5.0),
    'clf__max_features':      uniform(0.5, 0.4),
}


with mlflow.start_run():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_dist, n_iter=80, scoring='f1', cv=cv,
                    n_jobs=-1, random_state=42, refit=True)
    search.fit(X_train, y_train)

    best = search.best_estimator_

    # 5. Tune threshold on CV (not on test) to avoid leakage
    from sklearn.model_selection import cross_val_predict
    oof_prob = cross_val_predict(best, X_train, y_train, cv=cv,
                                method='predict_proba', n_jobs=-1)[:, 1]
    ts = np.linspace(0.1, 0.9, 81)
    best_t = max(ts, key=lambda t: f1_score(y_train, (oof_prob >= t).astype(int)))

    # 6. Evaluate
    tr_pred = (best.predict_proba(X_train)[:, 1] >= best_t).astype(int)
    te_pred = (best.predict_proba(X_test)[:, 1]  >= best_t).astype(int)
    f1_tr, f1_te = f1_score(y_train, tr_pred), f1_score(y_test, te_pred)

    # Log all parameter combinations and their mean test scores
    results = search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(search.best_params_)

    # Store and evaluate the best model
    best_model = search.best_estimator_

    train_report = classification_report(y_train, tr_pred, output_dict=True)
    test_report = classification_report(y_test, te_pred, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally
    model_path = "best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "cooldude101983/tourism-package-prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_model_v1.joblib",
        path_in_repo="best_tourism_package_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
