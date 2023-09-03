import argparse
import logging as logger
import warnings
import os
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.svm import SVC
from src.utils import Split
from optuna.integration import OptunaSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.features_preprocessing import Preprocessor

# Preset settings
RANDOM_STATE = 42
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.filterwarnings("ignore")
logger.getLogger().setLevel(logger.INFO)


class ClassifierModel:
    def __init__(
        self,
        df_path,
        target_path,
        n_samples=-1,
        n_trials=5,
        classifier_type: str = "logreg",
        save_model=False,
    ):
        self.df_path = df_path
        self.target_path = target_path
        self.n_samples = n_samples
        self.n_trials = n_trials

        self.classifier_type = classifier_type
        self.save_model = save_model
        self.skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        self.train_X = self.train_y = self.test_X = self.test_y = None,None,None,None

    def _training_setup(self):
        # Split
        self.train_X, self.train_y = Split(
            df_path=self.df_path, target_path=self.target_path
        ).get_train_data()

        self.test_X, self.test_y = Split(
            df_path=self.df_path, target_path=self.target_path
        ).get_test_data()

        # Model pipeline
        if self.classifier_type == "logreg":
            logreg_pipeline = make_pipeline(
                Preprocessor(tfidf_on=True, tf_idf_max_features=100),
                LogisticRegression(
                    random_state=RANDOM_STATE, class_weight={0: 50, 1: 50}
                ),
            )

            """init hyper-param"""
            param_distributions = {
                "logisticregression__C": optuna.distributions.CategoricalDistribution(
                    [1.0]
                ),
            }

            """optimization"""
            self.pipeline = OptunaSearchCV(
                logreg_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="recall",
            )

        elif self.classifier_type == "svc":
            svc_pipeline = make_pipeline(
                Preprocessor(tfidf_on=True, tf_idf_max_features=100),
                SVC(random_state=RANDOM_STATE, class_weight={0: 50, 1: 50}),
            )

            """init hyper-param"""
            param_distributions = {
                "logisticregression__C": optuna.distributions.CategoricalDistribution(
                    [1, 10, 50, 100, 200, 500]
                ),
            }

            """optimization"""
            self.pipeline = OptunaSearchCV(
                svc_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="recall",
            )

        elif self.classifier_type == "xgb":
            xgb_pipeline = make_pipeline(
                Preprocessor(tfidf_on=True, tf_idf_max_features=100),
                XGBClassifier(random_state=RANDOM_STATE),
            )

            """init hyper-param"""
            param_distributions = {
                "xgbclassifier__learning_rate": optuna.distributions.CategoricalDistribution(
                    [0.001, 0.005, 0.01, 0.05, 0.1]
                ),
                "xgbclassifier__max_depth": optuna.distributions.CategoricalDistribution(
                    [10, 8, 5]
                ),
                "xgbclassifier__min_child_weight": optuna.distributions.CategoricalDistribution(
                    [15, 13, 11]
                ),
                # imbalance param: subsample: 0-1
                "xgbclassifier__subsample": optuna.distributions.CategoricalDistribution(
                    [0.6, 0.7]
                ),
                "xgbclassifier__n_estimators": optuna.distributions.CategoricalDistribution(
                    [500, 600, 800, 1000]
                ),
                # imbalance param: scale_pos_weight :  sum(negative instances) / sum(positive instances)
                "xgbclassifier__scale_pos_weight": optuna.distributions.CategoricalDistribution(
                    [9.85]
                ),
                # imbalance param: max_delta_step : 1-10
                "xgbclassifier__max_delta_step": optuna.distributions.CategoricalDistribution(
                    [10]
                ),
            }

            """optimization"""
            self.pipeline = OptunaSearchCV(
                xgb_pipeline,
                param_distributions,
                cv=self.skf,
                n_trials=self.n_trials,
                random_state=RANDOM_STATE,
                verbose=0,
                scoring="roc_auc",
            )


    def train(self):
        """... and one method to rule them all. (c)"""
        self._training_setup()
        train_X, train_y = self.train_X, self.train_y

        """MLFlow Config"""
        logger.info("Setting up MLFlow Config")

        mlflow.set_experiment("Positive & Negative reviews classifier")

        # """Search for previous runs and get run_id if present"""
        # logger.info("Searching for previous runs for given model type")
        # df_runs = mlflow.search_runs(filter_string="tags.Model = '{0}'".format('XGB'))
        # df_runs = df_runs.loc[~df_runs['tags.Version'].isna(), :] if 'tags.Version' in df_runs else pd.DataFrame()
        # run_id = df_runs.loc[df_runs['tags.Version'] == run_version, 'run_id'].iloc[0]
        # run_id =3
        # load_prev = True
        # run_version = len(df_runs) + 1

        """Train model and save train metrics to mlflow"""
        with mlflow.start_run():
            mlflow.set_tag(
                "mlflow.runName", f"train_{mlflow.active_run().info.run_name}"
            )
            """Train & predict"""
            self.pipeline.fit(train_X, train_y)
            logger.info("train is done")
            logger.info("↓↓↓ TRAIN METRICS ↓↓↓")

            # """Get mean train & validation metrics"""
            # precision_scores, recall_scores, macro_f1, weighted_f1, auc, best_score = (
            #     [],
            #     [],
            #     [],
            #     [],
            #     [],
            #     [],
            # )
            # tn_train, tp_train, fp_train, fn_train = [], [], [], []

            # for i, (train_idx, valid_idx) in enumerate(
            #     self.skf.split(train_X, train_y)
            # ):
            #     # Train & validation indexes
            #     fold_train_x, fold_valid_x = (
            #         train_X.iloc[train_idx],
            #         train_X.iloc[valid_idx],
            #     )
            #     fold_train_y, fold_valid_y = (
            #         train_y.iloc[train_idx],
            #         train_y.iloc[valid_idx],
            #     )
            #     fold_pred_y_train = self.pipeline.predict(fold_train_x)
            #
            #     """classification report of train set"""
            #     df = pd.DataFrame(
            #         classification_report(
            #             y_true=fold_train_y,
            #             y_pred=fold_pred_y_train,
            #             output_dict=1,
            #             target_names=["non-toxic", "toxic"],
            #         )
            #     ).transpose()
            #
            #     # Precision
            #     precision_scores.append(np.round(df.loc["toxic", "precision"], 2))
            #     # Recall
            #     recall_scores.append(np.round(df.loc["toxic", "recall"], 2))
            #     # Macro f1
            #     macro_f1.append(np.round(df.loc["macro avg", "f1-score"], 2))
            #     # Weighted f1
            #     weighted_f1.append(np.round(df.loc["weighted avg", "f1-score"], 2))
            #     # Best Score
            #     best_score.append(self.pipeline.best_score_)
            #     # AUC
            #     auc.append(roc_auc_score(fold_train_y, fold_pred_y_train))
            #     # Confusion matrix
            #     conf_matrix = confusion_matrix(fold_train_y, fold_pred_y_train)
            #     tn_train.append(conf_matrix[0][0])
            #     tp_train.append(conf_matrix[1][1])
            #     fp_train.append(conf_matrix[0][1])
            #     fn_train.append(conf_matrix[1][0])
            #
            # # Compute the mean of the metrics
            # mean_precision = sum(precision_scores) / len(precision_scores)
            # mean_recall = sum(recall_scores) / len(recall_scores)
            # mean_macro_f1 = sum(macro_f1) / len(macro_f1)
            # mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
            # mean_auc_train = round((sum(auc) / len(auc)), 2)
            # # mean_best_score_train = sum(best_score) / len(best_score)
            # mean_tn_train = int(sum(tn_train) / len(tn_train))
            # mean_tp_train = int(sum(tp_train) / len(tp_train))
            # mean_fp_train = int(sum(fp_train) / len(fp_train))
            # mean_fn_train = int(sum(fn_train) / len(fn_train))
            #
            # """Show train metrics"""
            # # TODO: how to insert classification_report of cross validation?
            # #  I have no train_y, pred_y, and can't use mean
            # # logger.info(
            # #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
            # # )
            # logger.info(f"\n    Area Under the Curve score: {mean_auc_train}")
            # logger.info(
            #     f"\n Correlation matrix"
            #     f"\n (true negatives, false positives)\n (false negatives, true positives)"
            #     f"\n {mean_tn_train, mean_fp_train}\n {mean_fn_train, mean_tp_train}"
            # )
            # logger.info("Training Complete. Logging train results into MLFlow")
            #
            # """Log train metrics"""
            # # Precision
            # mlflow.log_metric("Precision", mean_precision)
            #
            # # Recall
            # mlflow.log_metric("Recall", mean_recall)
            #
            # # macro_f1
            # mlflow.log_metric("F1_macro", mean_macro_f1)
            #
            # # weighted_f1
            # mlflow.log_metric("F1_weighted", mean_weighted_f1)
            #
            # # Best Score
            # # mlflow.log_metric("Best Score", "%.2f " % mean_best_score_train)
            #
            # # AUC
            # mlflow.log_metric("AUC", mean_auc_train)
            #
            # # Confusion matrix
            # mlflow.log_metric("Conf_TN", mean_tn_train)
            # mlflow.log_metric("Conf_TP", mean_tp_train)
            # mlflow.log_metric("Conf_FP", mean_fp_train)
            # mlflow.log_metric("Conf_FN", mean_fn_train)
            #
            # """Log hyperparams"""
            # # best of hyperparameter tuning
            # mlflow.log_param(
            #     "Best Params",
            #     {k: round(v, 2) for k, v in self.pipeline.best_params_.items()},
            # )
            #
            # # number of input comments
            # mlflow.log_param("n_samples", self.n_samples)
            #
            # # number of trials of hyperparameter tuning
            # mlflow.log_param("n_trials", self.n_trials)
            #
            # """log model type"""
            # mlflow.set_tag("Model", self.classifier_type)

        # """Predict valid metrics and save to mlflow"""
        # with mlflow.start_run():
        #     mlflow.set_tag(
        #         "mlflow.runName", f"valid_{mlflow.active_run().info.run_name}"
        #     )
        #
        #     """Get mean validation metrics"""
        #     precision_scores, recall_scores, macro_f1, weighted_f1, auc, best_score = (
        #         [],
        #         [],
        #         [],
        #         [],
        #         [],
        #         [],
        #     )
        #     tn_valid, tp_valid, fp_valid, fn_valid = [], [], [], []
        #
        #     for i, (train_idx, valid_idx) in enumerate(
        #         self.skf.split(train_X, train_y)
        #     ):
        #         # Train & validation indexes
        #         fold_train_x, fold_valid_x = (
        #             train_X.iloc[train_idx],
        #             train_X.iloc[valid_idx],
        #         )
        #         fold_train_y, fold_valid_y = (
        #             train_y.iloc[train_idx],
        #             train_y.iloc[valid_idx],
        #         )
        #         fold_pred_y_valid = self.pipeline.predict(fold_valid_x)
        #
        #         """classification report of valid set"""
        #         df = pd.DataFrame(
        #             classification_report(
        #                 y_true=fold_valid_y,
        #                 y_pred=fold_pred_y_valid,
        #                 output_dict=1,
        #                 target_names=["non-toxic", "toxic"],
        #             )
        #         ).transpose()
        #
        #         # Precision
        #         precision_scores.append(np.round(df.loc["toxic", "precision"], 2))
        #         # Recall
        #         recall_scores.append(np.round(df.loc["toxic", "recall"], 2))
        #         # Macro f1
        #         macro_f1.append(np.round(df.loc["macro avg", "f1-score"], 2))
        #         # Weighted f1
        #         weighted_f1.append(np.round(df.loc["weighted avg", "f1-score"], 2))
        #         # Best Score
        #         best_score.append(self.pipeline.best_score_)
        #         # AUC
        #         auc.append(roc_auc_score(fold_valid_y, fold_pred_y_valid))
        #         # Confusion matrix
        #         conf_matrix = confusion_matrix(fold_valid_y, fold_pred_y_valid)
        #         tn_valid.append(conf_matrix[0][0])
        #         tp_valid.append(conf_matrix[1][1])
        #         fp_valid.append(conf_matrix[0][1])
        #         fn_valid.append(conf_matrix[1][0])
        #
        #     # Compute the mean of the metrics
        #     mean_precision = sum(precision_scores) / len(precision_scores)
        #     mean_recall = sum(recall_scores) / len(recall_scores)
        #     mean_macro_f1 = sum(macro_f1) / len(macro_f1)
        #     mean_weighted_f1 = sum(weighted_f1) / len(weighted_f1)
        #     mean_auc = round((sum(auc) / len(auc)), 2)
        #     # mean_best_score = sum(best_score) / len(best_score)
        #     mean_tn_valid = int(sum(tn_valid) / len(tn_valid))
        #     mean_tp_valid = int(sum(tp_valid) / len(tp_valid))
        #     mean_fp_valid = int(sum(fp_valid) / len(fp_valid))
        #     mean_fn_valid = int(sum(fn_valid) / len(fn_valid))
        #
        #     logger.info("↓↓↓ VALID METRICS ↓↓↓")
        #     """Show valid metrics"""
        #     # TODO: how to insert classification_report of cross validation?
        #     #  I have no train_y, pred_y, and can't use mean
        #     # logger.info(
        #     #     f"\n{pd.DataFrame(classification_report(train_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
        #     # )
        #     logger.info(f"\n    Area Under the Curve score: {mean_auc}")
        #     logger.info(
        #         f"\n Correlation matrix"
        #         f"\n (true negatives, false positives)\n (false negatives, true positives)"
        #         f"\n {mean_tn_valid, mean_fp_valid}\n {mean_fn_valid, mean_tp_valid}"
        #     )
        #     logger.info("Logging validation results into MLFlow")
        #
        #     """Log valid metrics"""
        #     # Precision
        #     mlflow.log_metric("Precision", mean_precision)
        #
        #     # Recall
        #     mlflow.log_metric("Recall", mean_recall)
        #
        #     # macro_f1
        #     mlflow.log_metric("F1_macro", mean_macro_f1)
        #
        #     # weighted_f1
        #     mlflow.log_metric("F1_weighted", mean_weighted_f1)
        #
        #     # # Best Score
        #     # mlflow.log_metric("Best Score", "%.2f " % mean_best_score)
        #
        #     # AUC
        #     mlflow.log_metric("AUC", mean_auc)
        #
        #     # Confusion matrix
        #     mlflow.log_metric("Conf_TN", mean_tn_valid)
        #     mlflow.log_metric("Conf_TP", mean_tp_valid)
        #     mlflow.log_metric("Conf_FP", mean_fp_valid)
        #     mlflow.log_metric("Conf_FN", mean_fn_valid)
        #
        # """Predict test metrics and save to mlflow"""
        # with mlflow.start_run():
        #     mlflow.set_tag(
        #         "mlflow.runName", f"test_{mlflow.active_run().info.run_name}"
        #     )
        #     test_x, test_y = self.test_X, self.test_y
        #     """Predict on test data"""
        #     pred_y = self.pipeline.predict(test_x)
        #     # self.pipeline.save()
        #
        #     """classification report of train set"""
        #     df = pd.DataFrame(
        #         classification_report(
        #             y_true=test_y,
        #             y_pred=pred_y,
        #             output_dict=1,
        #             target_names=["non-toxic", "toxic"],
        #         )
        #     ).transpose()
        #
        #     """Show test metrics"""
        #     logger.info("↓↓↓ TEST METRICS ↓↓↓")
        #     logger.info(
        #         f"\n{pd.DataFrame(classification_report(test_y, pred_y, output_dict=1, target_names=['non-toxic', 'toxic'])).transpose()}"
        #     )
        #     logger.info(
        #         f"\n    Area Under the Curve score: {round(roc_auc_score(test_y, pred_y), 2)}"
        #     )
        #     logger.info(
        #         f"\n [true negatives  false positives]\n [false negatives  true positives] \
        #                 \n {confusion_matrix(test_y, pred_y)}"
        #     )
        #
        #     # logger.info(" Logging results into file_log.log")
        #     logger.info("Logging test results into MLFlow")
        #
        #     """Log train metrics"""
        #     # Precisionpycharm
        #     mlflow.log_metric("Precision", np.round(df.loc["toxic", "precision"], 2))
        #
        #     # Recall
        #     mlflow.log_metric("Recall", np.round(df.loc["toxic", "recall"], 2))
        #
        #     # macro_f1
        #     mlflow.log_metric("F1_macro", np.round(df.loc["macro avg", "f1-score"], 2))
        #
        #     # weighted_f1
        #     mlflow.log_metric(
        #         "F1_weighted", np.round(df.loc["weighted avg", "f1-score"], 2)
        #     )
        #
        #     # Best Score
        #     # mlflow.log_metric("Best Score", "%.2f " % self.pipeline.best_score_)
        #
        #     # AUC
        #     mlflow.log_metric("AUC", round(roc_auc_score(test_y, pred_y), 2))
        #
        #     # Confusion matrix
        #     conf_matrix = confusion_matrix(test_y, pred_y)
        #     mlflow.log_metric("Conf_TN", conf_matrix[0][0])
        #     mlflow.log_metric("Conf_TP", conf_matrix[1][1])
        #     mlflow.log_metric("Conf_FP", conf_matrix[0][1])
        #     mlflow.log_metric("Conf_FN", conf_matrix[1][0])
        #
        # if self.save_model:
        #     try:
        #         os.mkdir("./data/")
        #     except:
        #         pass
        #     """Log(save) model"""
        #     import pickle
        #
        #     # Define the path to save the model
        #     model_path = "data/classic_model.pkl"
        #     # Save the model to a pickle file
        #     with open(model_path, "wb") as model_file:
        #         pickle.dump(self.pipeline, model_file)
        #     logger.info("Model Trained and saved into MLFlow artifact location")
        # else:
        #     logger.info("Model Trained but not saved into MLFlow artifact location")


if __name__ == "__main__":
    # Import train dataset
    # url = "https://drive.google.com/file/d/1eEtlmdLUTZnyY34g9bL5D3XuWLzSEqBU/view?usp=sharing"
    # path = "https://drive.google.com/uc?id=" + url.split("/")[-2]
    # df1 = pd.read_csv(path)
    path = "D:/Programming/DB's/Positive_negative_reviews_classification/reviews.csv"
    # url = "https://drive.google.com/file/d/1x2Tdn1UGhQ6x08yfchUykUJvid257vFY/view?usp=sharing"
    # path2 = "https://drive.google.com/uc?id=" + url.split("/")[-2]
    # _target = pd.read_csv(path2)
    target = "D:/Programming/DB's/Positive_negative_reviews_classification/labels.csv"

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--df_path", help="Reviews path", default=path)
    parser.add_argument(
        "--target_path", help="Target path", default=target
    )
    parser.add_argument("--n_samples", help="How many samples to pass?", default=-1)
    parser.add_argument(
        "--n_trials", help="How many trials for hyperparameter tuning?", default=1
    )
    parser.add_argument(
        "--classifier_type",
        help='Choose "logreg", "svc" or "xgb"',
        default="logreg",
    )
    parser.add_argument(
        "--save_model",
        help="Choose True or False",
        default=False,
    )
    args = parser.parse_args()

    classifier = ClassifierModel(
        df_path=args.df_path,
        target_path=args.target_path,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        classifier_type=args.classifier_type,
        save_model=args.save_model,
    )
    classifier.train()
    # print(classifier._training_setup())
    # print(Split(df_path=path, target_path=target).get_train_data())