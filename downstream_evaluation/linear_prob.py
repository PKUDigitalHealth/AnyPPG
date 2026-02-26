import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

BOOTSTRAP_ROUNDS = 500
CV_FOLDS = 5
RANDOM_STATE = 42


def load_npz(path):
    data = np.load(path)
    return data["embeds"], data["labels"].reshape(-1)

def calculate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1 and np.std(y_pred) > 1e-9:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = 0
    
    var = np.var(y_true)
    pmse = (mse / var) if var > 1e-9 else 0
    
    return {
        "MAE": mae, "MSE": mse, "RMSE": rmse, 
        "R2": r2, "Pearson_R": r, "PMSE": pmse
    }

def calculate_classification_metrics(y_true, y_pred, y_prob, n_classes):
    res = {}
    res["ACC"] = accuracy_score(y_true, y_pred)
    res["F1_Macro"] = f1_score(y_true, y_pred, average="macro")
    
    if n_classes == 2:
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            res["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            res["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
            res["Pos_Samples"] = int(tp + fn)
            res["Neg_Samples"] = int(tn + fp)
            
            if y_prob is not None:
                res["AUROC"] = roc_auc_score(y_true, y_prob[:, 1])
                res["AUPRC"] = average_precision_score(y_true, y_prob[:, 1])
            else:
                res["AUROC"] = np.nan
                res["AUPRC"] = np.nan
        except Exception as e:
            res["Sensitivity"] = 0
            res["Specificity"] = 0
            res["AUROC"] = np.nan
    else:
        try:
            if y_prob is not None:
                res["AUROC_OVR"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            else:
                res["AUROC_OVR"] = np.nan
        except:
            res["AUROC_OVR"] = np.nan
        res["Sensitivity"] = "N/A"
        res["Specificity"] = "N/A"
        
    return res

def bootstrap_ci(y_true, y_pred, y_prob, metric_func, n_classes, task_type):
    stats = {}
    for _ in range(BOOTSTRAP_ROUNDS):
        indices = resample(np.arange(len(y_true)), replace=True)
        
        if task_type == "classification":
            if len(np.unique(y_true[indices])) < 2:
                continue
            
        if task_type == "regression":
            m = metric_func(y_true[indices], y_pred[indices])
        else:
            curr_prob = y_prob[indices] if y_prob is not None else None
            m = metric_func(y_true[indices], y_pred[indices], curr_prob, n_classes)
            
        for k, v in m.items():
            if k not in stats: stats[k] = []
            if isinstance(v, (int, float)) and not np.isnan(v):
                stats[k].append(v)
                
    ci_results = {}
    for k, v_list in stats.items():
        if not v_list: 
            ci_results[k] = "N/A"
            continue
        mean = np.mean(v_list)
        lower = np.percentile(v_list, 2.5)
        upper = np.percentile(v_list, 97.5)
        ci_results[k] = f"{mean:.4f} ({lower:.4f}-{upper:.4f})"
        
    return ci_results


def run_linear_probe_solver(data_dir, task_type="classification"):
    try:
        X_train, y_train = load_npz(data_dir / "train_embeds.npz")
        X_test, y_test = load_npz(data_dir / "test_embeds.npz")
    except FileNotFoundError:
        return None, None

    print(f"    -> Data Loaded: Train={len(y_train)}, Test={len(y_test)}")

    if task_type == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(le.classes_)
        
        if n_classes > 2:
            pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=RANDOM_STATE))])
            param_grid = {
                "rf__n_estimators": [100, 200],
                "rf__max_depth": [10, 20],
                "rf__min_samples_split": [2, 5]
            }
            scoring = "accuracy"
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs", max_iter=1000))])
            param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
            scoring = "roc_auc"
    else:
        n_classes = 0
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", Ridge())])
        param_grid = {"reg__alpha": [0.1, 1.0, 10.0, 100.0]}
        scoring = "neg_mean_absolute_error"

    search = GridSearchCV(pipe, param_grid, cv=CV_FOLDS, scoring=scoring, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    best_param_str = str(search.best_params_)
    print(f"    -> Tuning Done. Best Params: {best_param_str}")
    
    y_pred = best_model.predict(X_test)
    y_prob = None
    
    if task_type == "classification":
        y_prob = best_model.predict_proba(X_test)
        metrics = calculate_classification_metrics(y_test, y_pred, y_prob, n_classes)
        print("    -> Running Bootstrap CI...")
        cis = bootstrap_ci(y_test, y_pred, y_prob, calculate_classification_metrics, n_classes, task_type)
        
        main_score = metrics.get('AUROC', metrics.get('AUROC_OVR', 0))
        print(f"    -> [RESULT] AUROC: {main_score:.4f} | ACC: {metrics['ACC']:.4f}")
        
    else:
        metrics = calculate_regression_metrics(y_test, y_pred)
        print("    -> Running Bootstrap CI...")
        cis = bootstrap_ci(y_test, y_pred, None, calculate_regression_metrics, 0, task_type)
        
        main_score = metrics['MAE']
        print(f"    -> [RESULT] MAE: {main_score:.4f} | R2: {metrics['R2']:.4f}")
    
    final_report = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
    for k, v in cis.items():
        final_report[f"{k}_CI"] = v
        
    final_report["Best_Params"] = best_param_str
    
    raw_predictions = {
        "y_true": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob
    }
    
    return final_report, raw_predictions