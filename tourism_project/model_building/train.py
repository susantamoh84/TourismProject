# for data manipulation
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/cooldude101983/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/cooldude101983/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/cooldude101983/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/cooldude101983/tourism-package-prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test  = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)['ProdTaken']
y_test  = pd.read_csv(ytest_path)['ProdTaken']

# ----------------------------------------------------------------------
# 2. Cleaning + feature engineering
#    Row-wise transforms only -> safe to apply to train and test separately.
# ----------------------------------------------------------------------
def engineer(df):
    df = df.copy()
 
    # --- fix dirty categorical labels (idempotent) ---
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
 
    # --- ordinal encodings for naturally-ordered categories ---
    desig_rank = {'Executive': 1, 'Manager': 2, 'Senior Manager': 3, 'AVP': 4, 'VP': 5}
    prod_rank  = {'Basic': 1, 'Standard': 2, 'Deluxe': 3, 'Super Deluxe': 4, 'King': 5}
    df['DesignationRank'] = df['Designation'].map(desig_rank)
    df['ProductRank']     = df['ProductPitched'].map(prod_rank)
 
    # --- engineered interaction / ratio features ---
    df['TotalVisitors']       = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
    df['IncomePerVisitor']    = df['MonthlyIncome'] / (df['TotalVisitors'] + 1)
    df['EngagementScore']     = df['NumberOfFollowups'] * df['PitchSatisfactionScore']
    df['PitchPerTrip']        = df['DurationOfPitch'] / (df['NumberOfTrips'] + 1)
    df['ProductVsDesignation'] = df['ProductRank'] - df['DesignationRank']  # upsell signal
    df['IncomeXSeniority']    = df['MonthlyIncome'] * df['DesignationRank']
 
    # --- life-stage bins ---
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100],
                          labels=['young', 'mid', 'senior', 'older']).astype(str)
    return df
 
X_train = engineer(X_train)
X_test  = engineer(X_test)
 
cat_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
num_cols = X_train.select_dtypes(exclude=['object', 'string']).columns.tolist()
 
# ----------------------------------------------------------------------
# 3. Preprocessing
#    Trees don't need scaling -> passthrough numerics; one-hot the categoricals.
# ----------------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
])
 
# scale_pos_weight handles the ~19% / 81% class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
 
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )),
])
 
# ----------------------------------------------------------------------
# 4. Hyperparameter tuning (RandomizedSearchCV, regularization-focused)
# ----------------------------------------------------------------------
param_distributions = {
    'classifier__n_estimators':     randint(100, 450),
    'classifier__learning_rate':    uniform(0.01, 0.12),
    'classifier__max_depth':        randint(2, 6),
    'classifier__min_child_weight': randint(3, 25),   # ~min leaf size; fights overfitting
    'classifier__subsample':        uniform(0.6, 0.4),
    'classifier__colsample_bytree': uniform(0.5, 0.5),
    'classifier__gamma':            uniform(0.0, 5.0), # min loss reduction to split
    'classifier__reg_alpha':        uniform(0.0, 5.0), # L1
    'classifier__reg_lambda':       uniform(1.0, 9.0), # L2
}

with mlflow.start_run():
  
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
  
  search = RandomizedSearchCV(
      estimator=pipeline,
      param_distributions=param_distributions,
      n_iter=80,
      scoring='f1',
      cv=cv,
      n_jobs=-1,
      random_state=RANDOM_STATE,
      refit=True,
      verbose=1,
  )
  
  search.fit(X_train, y_train)
  best_model = search.best_estimator_
  
  print("\nBest hyperparameters:")
  for k, v in search.best_params_.items():
      print(f"  {k}: {v}")
  print(f"Best CV f1: {search.best_score_:.4f}")

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
  
  # ----------------------------------------------------------------------
  # 5. (Optional) leakage-free threshold tuning
  #    Choose the probability threshold on TRAIN via cross-validation,
  #    never on the test set, then apply it to test.
  # ----------------------------------------------------------------------
  oof_proba = cross_val_predict(best_model, X_train, y_train, cv=cv,
                                method='predict_proba', n_jobs=-1)[:, 1]
  thresholds = np.linspace(0.1, 0.9, 81)
  best_threshold = max(thresholds,
                      key=lambda t: f1_score(y_train, (oof_proba >= t).astype(int)))
  print(f"\nTuned threshold (from train CV): {best_threshold:.3f}")
  
  # ----------------------------------------------------------------------
  # 6. Evaluate on the held-out test set
  # ----------------------------------------------------------------------
  train_proba = best_model.predict_proba(X_train)[:, 1]
  test_proba  = best_model.predict_proba(X_test)[:, 1]
  
  for label, thr in [("default 0.50", 0.50), (f"tuned {best_threshold:.3f}", best_threshold)]:
      tr_f1 = f1_score(y_train, (train_proba >= thr).astype(int))
      te_f1 = f1_score(y_test,  (test_proba  >= thr).astype(int))
      print(f"\n--- Threshold: {label} ---")
      print(f"Train F1: {tr_f1:.4f} | Test F1: {te_f1:.4f} | Gap: {tr_f1 - te_f1:.4f}")
  
  tr_pred = (train_proba >= 0.5).astype(int)
  te_pred = (test_proba >= 0.5).astype(int)
  print("\nTest classification report:")
  train_report = classification_report(y_train, tr_pred, output_dict=True)
  test_report = classification_report(y_test, te_pred, output_dict=True)
  print(train_report)
  print(test_report)

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
