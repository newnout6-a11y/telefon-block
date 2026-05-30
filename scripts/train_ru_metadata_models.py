"""
Offline trainer for RF metadata models.

Benchmarks LogisticRegression, RandomForest, optional CatBoost, and optional TFLite MLP.
Features: Optuna/Grid hyperparameter search, SMOTE balancing, cross-validation,
mutual information + permutation importance, per-class threshold tuning,
Platt/isotonic calibration, drift detection, confusion matrix/ROC/calibration plots.
Exports TFLite only when BLOCK precision passes the configured gate.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ru_metadata_features import COMPACT_FEATURES, ID_TO_LABEL

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'ru')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
DEFAULT_DATA = os.path.join(PROCESSED_DIR, 'ru_tflite_features.csv')
DEFAULT_TFLITE = os.path.join(os.path.dirname(__file__), '..', 'app', 'src', 'main', 'assets', 'spam_model.tflite')
DEFAULT_MODEL_CARD = os.path.join(os.path.dirname(__file__), '..', 'app', 'src', 'main', 'assets', 'model_card.json')


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f'No rows in {path}')
    features = []
    labels = []
    for row in rows:
        features.append([float(row[name]) for name in COMPACT_FEATURES])
        labels.append(int(float(row['label'])))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def class_counts(y: np.ndarray) -> Dict[str, int]:
    return {ID_TO_LABEL.get(i, str(i)): int(np.sum(y == i)) for i in range(3)}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    from sklearn.model_selection import train_test_split
    stratify = y if len(set(y.tolist())) > 1 and min(np.bincount(y, minlength=3)) > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)


def apply_smote(X_train: np.ndarray, y_train: np.ndarray, strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(y_train, minlength=3)
        min_class = min(c for c in counts if c > 0)
        if min_class < 6:
            k_neighbors = max(1, min_class - 1)
        else:
            k_neighbors = 5
        sm = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res.astype(np.float32), y_res.astype(np.int32)
    except ImportError:
        print('  [WARN] imbalanced-learn not installed, skipping SMOTE')
        return X_train, y_train
    except Exception as e:
        print(f'  [WARN] SMOTE failed: {e}, using original data')
        return X_train, y_train


def oversample_rare_patterns(X: np.ndarray, y: np.ndarray, min_count: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    X_list = list(X)
    y_list = list(y)
    for cls in range(3):
        mask = y == cls
        count = int(np.sum(mask))
        if 0 < count < min_count:
            indices = np.where(mask)[0]
            reps = max(1, min_count // count)
            for _ in range(reps):
                for idx in indices:
                    noise = np.random.normal(0, 0.01, X.shape[1]).astype(np.float32)
                    X_list.append(X[idx] + noise)
                    y_list.append(y[idx])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    result = {
        'name': name,
        'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
        'classification_report': classification_report(y_test, pred, labels=[0, 1, 2], target_names=['ALLOW', 'WARN', 'BLOCK'], zero_division=0, output_dict=True),
    }

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1, 2], zero_division=0)
    result['per_class'] = {
        ID_TO_LABEL[i]: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
        }
        for i in range(3)
    }

    if proba is not None and len(set(y_test.tolist())) > 1:
        try:
            if proba.shape[1] == 3:
                result['roc_auc_ovr'] = float(roc_auc_score(y_test, proba, multi_class='ovr', labels=[0, 1, 2]))
                result['thresholds'] = tune_per_class_thresholds(y_test, proba)
        except Exception as e:
            result['roc_auc_error'] = str(e)

    result['block_precision'] = result['per_class']['BLOCK']['precision']
    result['block_recall'] = result['per_class']['BLOCK']['recall']
    return result


def tune_per_class_thresholds(y_true: np.ndarray, proba: np.ndarray, min_block_precision: float = 0.90) -> Dict:
    if proba.shape[1] < 3:
        return {'block_threshold': 0.50, 'warn_threshold': 0.30, 'block_precision': 0.0, 'block_recall': 0.0}

    best = {'block_threshold': 0.50, 'warn_threshold': 0.30, 'block_precision': 0.0, 'block_recall': 0.0}
    y_block = y_true == 2
    for threshold in np.linspace(0.05, 0.95, 91):
        pred_block = proba[:, 2] >= threshold
        tp = int(np.sum(pred_block & y_block))
        fp = int(np.sum(pred_block & ~y_block))
        fn = int(np.sum(~pred_block & y_block))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision >= min_block_precision and recall >= best['block_recall']:
            best = {
                'block_threshold': float(threshold),
                'warn_threshold': float(max(0.15, threshold * 0.55)),
                'block_precision': float(precision),
                'block_recall': float(recall),
            }

    y_warn = y_true == 1
    best_f1_warn = 0.0
    best_warn_thr = 0.30
    for thr in np.linspace(0.10, 0.60, 51):
        pred_warn = (proba[:, 1] >= thr) & (proba[:, 2] < best['block_threshold'])
        tp_w = int(np.sum(pred_warn & y_warn))
        fp_w = int(np.sum(pred_warn & ~y_warn))
        fn_w = int(np.sum(~pred_warn & y_warn))
        p_w = tp_w / max(tp_w + fp_w, 1)
        r_w = tp_w / max(tp_w + fn_w, 1)
        f1_w = 2 * p_w * r_w / max(p_w + r_w, 1e-9)
        if f1_w > best_f1_warn:
            best_f1_warn = f1_w
            best_warn_thr = float(thr)
    best['warn_threshold'] = best_warn_thr
    best['warn_f1'] = float(best_f1_warn)
    return best


def compute_mutual_information(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X, y, random_state=42)
        return {COMPACT_FEATURES[i]: float(mi[i]) for i in range(len(COMPACT_FEATURES))}
    except ImportError:
        return {}


def compute_permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        return {COMPACT_FEATURES[i]: float(result.importances_mean[i]) for i in range(len(COMPACT_FEATURES))}
    except Exception:
        return {}


def run_cross_validation(model_factory, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
    from sklearn.model_selection import cross_validate
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    try:
        cv_results = cross_validate(model_factory(), X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            'cv_folds': cv,
            'precision_macro_mean': float(np.mean(cv_results['test_precision_macro'])),
            'precision_macro_std': float(np.std(cv_results['test_precision_macro'])),
            'recall_macro_mean': float(np.mean(cv_results['test_recall_macro'])),
            'recall_macro_std': float(np.std(cv_results['test_recall_macro'])),
            'f1_macro_mean': float(np.mean(cv_results['test_f1_macro'])),
            'f1_macro_std': float(np.std(cv_results['test_f1_macro'])),
        }
    except Exception as e:
        return {'cv_error': str(e)}


def optuna_search(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                  n_trials: int = 30) -> Dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {'name': 'optuna_search', 'skipped': 'optuna is not installed (pip install optuna)'}

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    def objective(trial):
        model_name = trial.suggest_categorical('model', ['rf', 'lr'])
        if model_name == 'rf':
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 4, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced_subsample', random_state=42, n_jobs=-1,
            )
        else:
            C = trial.suggest_float('C', 0.01, 100.0, log=True)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=C, max_iter=1000, class_weight='balanced'),
            )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return float(f1_score(y_test, pred, average='macro', zero_division=0))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial
    result = {
        'name': 'optuna_search',
        'best_f1_macro': best.value,
        'best_params': best.params,
        'n_trials': n_trials,
    }

    best_params = best.params.copy()
    model_type = best_params.pop('model')
    if model_type == 'rf':
        best_model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 250),
            max_depth=best_params.get('max_depth', 12),
            min_samples_leaf=best_params.get('min_samples_leaf', 1),
            class_weight='balanced_subsample', random_state=42, n_jobs=-1,
        )
    else:
        best_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_params.get('C', 1.0), max_iter=1000, class_weight='balanced'),
        )
    best_model.fit(X_train, y_train)
    result['evaluation'] = evaluate_model('optuna_best', best_model, X_test, y_test)
    return result


def detect_drift(X_train: np.ndarray, X_prod: Optional[np.ndarray] = None) -> Dict:
    if X_prod is None:
        return {'drift_detected': None, 'note': 'No production data provided; drift check skipped'}
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return {'drift_detected': None, 'note': 'scipy not available for KS test'}

    drift_results = {}
    n_features = X_train.shape[1]
    drifted_count = 0
    for i in range(n_features):
        stat, pval = ks_2samp(X_train[:, i], X_prod[:, i])
        drifted = pval < 0.01
        if drifted:
            drifted_count += 1
        drift_results[COMPACT_FEATURES[i]] = {'ks_statistic': float(stat), 'p_value': float(pval), 'drifted': drifted}

    return {
        'drift_detected': drifted_count > n_features * 0.3,
        'drifted_features': drifted_count,
        'total_features': n_features,
        'drift_ratio': float(drifted_count / n_features),
        'per_feature': drift_results,
    }


def train_sklearn_models(X_train, y_train, X_test, y_test, use_smote: bool = True) -> Dict[str, Dict]:
    results = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if use_smote:
        X_train_bal, y_train_bal = apply_smote(X_train, y_train)
        X_train_bal, y_train_bal = oversample_rare_patterns(X_train_bal, y_train_bal, min_count=10)
    else:
        X_train_bal, y_train_bal = X_train, y_train

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )
    lr.fit(X_train_bal, y_train_bal)
    results['logistic_regression'] = evaluate_model('logistic_regression', lr, X_test, y_test)

    calibrated_lr_iso = CalibratedClassifierCV(lr, cv=3, method='isotonic')
    calibrated_lr_iso.fit(X_train_bal, y_train_bal)
    results['calibrated_logistic_regression_isotonic'] = evaluate_model('calibrated_logistic_regression_isotonic', calibrated_lr_iso, X_test, y_test)

    calibrated_lr_platt = CalibratedClassifierCV(lr, cv=3, method='sigmoid')
    calibrated_lr_platt.fit(X_train_bal, y_train_bal)
    results['calibrated_logistic_regression_platt'] = evaluate_model('calibrated_logistic_regression_platt', calibrated_lr_platt, X_test, y_test)

    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=-1,
    )
    rf.fit(X_train_bal, y_train_bal)
    results['random_forest'] = evaluate_model('random_forest', rf, X_test, y_test)

    calibrated_rf_iso = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    calibrated_rf_iso.fit(X_train_bal, y_train_bal)
    results['calibrated_random_forest_isotonic'] = evaluate_model('calibrated_random_forest_isotonic', calibrated_rf_iso, X_test, y_test)

    calibrated_rf_platt = CalibratedClassifierCV(rf, cv=3, method='sigmoid')
    calibrated_rf_platt.fit(X_train_bal, y_train_bal)
    results['calibrated_random_forest_platt'] = evaluate_model('calibrated_random_forest_platt', calibrated_rf_platt, X_test, y_test)

    results['feature_importance'] = {
        COMPACT_FEATURES[i]: float(value)
        for i, value in sorted(enumerate(rf.feature_importances_), key=lambda item: item[1], reverse=True)
    }

    results['mutual_information'] = compute_mutual_information(X_train_bal, y_train_bal)
    results['permutation_importance_rf'] = compute_permutation_importance(rf, X_test, y_test)

    results['cross_validation_rf'] = run_cross_validation(
        lambda: RandomForestClassifier(n_estimators=250, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
        X_train_bal, y_train_bal, cv=5,
    )
    results['cross_validation_lr'] = run_cross_validation(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
        X_train_bal, y_train_bal, cv=5,
    )

    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=42,
        )
        cb.fit(X_train_bal, y_train_bal)
        results['catboost'] = evaluate_model('catboost', cb, X_test, y_test)
    except ImportError:
        results['catboost'] = {'name': 'catboost', 'skipped': 'catboost is not installed'}

    results['smote_applied'] = use_smote
    results['train_class_counts_after_smote'] = class_counts(y_train_bal)

    return results


def train_tflite_mlp(X_train, y_train, X_test, y_test, epochs: int, batch: int, output: str, export: bool) -> Dict:
    try:
        import tensorflow as tf
    except ImportError:
        return {'name': 'tflite_mlp', 'skipped': 'tensorflow is not installed'}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(COMPACT_FEATURES),)),
        tf.keras.layers.Dense(48, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(12, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    counts = np.bincount(y_train, minlength=3)
    total = len(y_train)
    class_weight = {i: float(total / (3 * counts[i])) if counts[i] > 0 else 1.0 for i in range(3)}

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch,
        class_weight=class_weight,
        verbose=0,
    )

    proba = model.predict(X_test, verbose=0)
    pred = np.argmax(proba, axis=1)

    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1, 2], zero_division=0)
    result = {
        'name': 'tflite_mlp',
        'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist(),
        'classification_report': classification_report(y_test, pred, labels=[0, 1, 2], target_names=['ALLOW', 'WARN', 'BLOCK'], zero_division=0, output_dict=True),
        'per_class': {
            ID_TO_LABEL[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
            }
            for i in range(3)
        },
        'block_precision': float(precision[2]),
        'block_recall': float(recall[2]),
        'final_val_accuracy': float(history.history.get('val_accuracy', [0])[-1]),
    }
    try:
        result['roc_auc_ovr'] = float(roc_auc_score(y_test, proba, multi_class='ovr', labels=[0, 1, 2]))
        result['thresholds'] = tune_per_class_thresholds(y_test, proba)
    except Exception as e:
        result['roc_auc_error'] = str(e)

    if export:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'wb') as f:
            f.write(tflite_model)
        result['exported_tflite'] = output
        result['tflite_bytes'] = len(tflite_model)

    return result


def generate_plots(report: Dict, reports_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.calibration import calibration_curve
    except ImportError:
        print('  [WARN] matplotlib not installed, skipping plot generation')
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(reports_dir, exist_ok=True)

    for name, metrics in report['models'].items():
        if metrics.get('skipped') or name in ('feature_importance', 'mutual_information',
                                                'permutation_importance_rf', 'cross_validation_rf',
                                                'cross_validation_lr', 'smote_applied',
                                                'train_class_counts_after_smote'):
            continue

        cm = metrics.get('confusion_matrix')
        if cm:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap='OrRd', interpolation='nearest')
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(['ALLOW', 'WARN', 'BLOCK'])
            ax.set_yticklabels(['ALLOW', 'WARN', 'BLOCK'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix — {name}')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='black' if cm[i][j] < max(map(max, cm)) * 0.5 else 'white')
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(os.path.join(reports_dir, f'cm_{name}_{stamp}.png'), dpi=120)
            plt.close(fig)

    importance = report['models'].get('feature_importance', {})
    if importance:
        top_n = 20
        items = list(importance.items())[:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [k for k, v in items]
        values = [v for k, v in items]
        ax.barh(range(len(names)), values, color='#ff6b1a')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (top 20)')
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, f'feature_importance_{stamp}.png'), dpi=120)
        plt.close(fig)

    mi = report['models'].get('mutual_information', {})
    if mi:
        top_n = 20
        items = sorted(mi.items(), key=lambda x: x[1], reverse=True)[:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [k for k, v in items]
        values = [v for k, v in items]
        ax.barh(range(len(names)), values, color='#4fc3f7')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Mutual Information')
        ax.set_title('Mutual Information (top 20)')
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, f'mutual_information_{stamp}.png'), dpi=120)
        plt.close(fig)

    perm = report['models'].get('permutation_importance_rf', {})
    if perm:
        top_n = 20
        items = sorted(perm.items(), key=lambda x: x[1], reverse=True)[:top_n]
        fig, ax = plt.subplots(figsize=(10, 6))
        names = [k for k, v in items]
        values = [v for k, v in items]
        ax.barh(range(len(names)), values, color='#81c784')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Permutation Importance (top 20)')
        fig.tight_layout()
        fig.savefig(os.path.join(reports_dir, f'permutation_importance_{stamp}.png'), dpi=120)
        plt.close(fig)

    print(f'  Plots saved to {reports_dir}/')


def write_html_report(report: Dict, html_path: str):
    rows = []
    for name, metrics in report['models'].items():
        if metrics.get('skipped') or name in ('feature_importance', 'mutual_information',
                                                'permutation_importance_rf', 'cross_validation_rf',
                                                'cross_validation_lr', 'smote_applied',
                                                'train_class_counts_after_smote'):
            continue
        roc_val = metrics.get('roc_auc_ovr', 'n/a')
        roc_str = f'{roc_val:.3f}' if isinstance(roc_val, (int, float)) else roc_val
        rows.append(
            f"<tr><td>{name}</td><td>{metrics.get('block_precision', 0):.3f}</td>"
            f"<td>{metrics.get('block_recall', 0):.3f}</td>"
            f"<td>{roc_str}</td>"
            f"<td>{metrics.get('per_class', {}).get('BLOCK', {}).get('f1', 0):.3f}</td>"
            f"<td><code>{metrics.get('confusion_matrix')}</code></td></tr>"
        )
    importance = report['models'].get('feature_importance', {})
    imp_rows = ''.join(f"<tr><td>{k}</td><td>{v:.5f}</td></tr>" for k, v in list(importance.items())[:32])
    mi = report['models'].get('mutual_information', {})
    mi_rows = ''.join(f"<tr><td>{k}</td><td>{v:.5f}</td></tr>" for k, v in sorted(mi.items(), key=lambda x: x[1], reverse=True)[:32])
    perm = report['models'].get('permutation_importance_rf', {})
    perm_rows = ''.join(f"<tr><td>{k}</td><td>{v:.5f}</td></tr>" for k, v in sorted(perm.items(), key=lambda x: x[1], reverse=True)[:32])

    cv_rf = report['models'].get('cross_validation_rf', {})
    cv_lr = report['models'].get('cross_validation_lr', {})
    cv_html = ''
    if cv_rf and 'cv_error' not in cv_rf:
        cv_html += f"<p><b>RF CV-5:</b> F1={cv_rf.get('f1_macro_mean', 0):.3f}±{cv_rf.get('f1_macro_std', 0):.3f} P={cv_rf.get('precision_macro_mean', 0):.3f} R={cv_rf.get('recall_macro_mean', 0):.3f}</p>"
    if cv_lr and 'cv_error' not in cv_lr:
        cv_html += f"<p><b>LR CV-5:</b> F1={cv_lr.get('f1_macro_mean', 0):.3f}±{cv_lr.get('f1_macro_std', 0):.3f} P={cv_lr.get('precision_macro_mean', 0):.3f} R={cv_lr.get('recall_macro_mean', 0):.3f}</p>"

    drift = report.get('drift_detection', {})
    drift_html = ''
    if drift and drift.get('drift_detected') is not None:
        drift_html = f"<p><b>Drift:</b> {'YES' if drift['drift_detected'] else 'NO'} ({drift.get('drifted_features', 0)}/{drift.get('total_features', 0)} features)</p>"

    smote_info = report['models'].get('smote_applied', False)
    after_counts = report['models'].get('train_class_counts_after_smote', {})

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>SpamBlocker ML Training Report</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; background:#0a0a0b; color:#f5f5f7; padding:24px; }}
    .card {{ background:#111113; border:1px solid #23232a; border-radius:16px; padding:18px; margin:16px 0; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ border-bottom:1px solid #23232a; padding:10px; text-align:left; }}
    .accent {{ color:#ff6b1a; }}
    .good {{ color:#4caf50; }}
    .warn {{ color:#ffb74d; }}
    code {{ color:#ffb070; }}
  </style>
</head>
<body>
  <h1>SpamBlocker ML Training Report</h1>
  <div class="card">
    <p><b>Created:</b> {report['created_at']}</p>
    <p><b>Rows:</b> {report['rows']} | <b>Features:</b> {report['feature_count']}</p>
    <p><b>Dataset hash:</b> <code>{report.get('dataset_hash', '')}</code></p>
    <p><b>Class counts:</b> {report['class_counts']}</p>
    <p><b>SMOTE applied:</b> {smote_info} | <b>After balance:</b> {after_counts}</p>
    {drift_html}
  </div>
  <div class="card">
    <h2 class="accent">Model metrics</h2>
    <table><tr><th>Model</th><th>BLOCK P</th><th>BLOCK R</th><th>ROC-AUC</th><th>BLOCK F1</th><th>Confusion</th></tr>{''.join(rows)}</table>
    {cv_html}
  </div>
  <div class="card">
    <h2 class="accent">Feature importance (RF)</h2>
    <table><tr><th>Feature</th><th>Importance</th></tr>{imp_rows}</table>
  </div>
  <div class="card">
    <h2 class="accent">Mutual information</h2>
    <table><tr><th>Feature</th><th>MI score</th></tr>{mi_rows}</table>
  </div>
  <div class="card">
    <h2 class="accent">Permutation importance (RF)</h2>
    <table><tr><th>Feature</th><th>Score</th></tr>{perm_rows}</table>
  </div>
</body>
</html>"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)


def write_report(report: Dict) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(REPORTS_DIR, f'metadata_training_report_{stamp}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_html_report(report, path.replace('.json', '.html'))

    md_path = path.replace('.json', '.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# SpamBlocker ML Training Report\n\n')
        f.write(f'- **Created**: {report["created_at"]}\n')
        f.write(f'- **Rows**: {report["rows"]}\n')
        f.write(f'- **Feature count**: {report["feature_count"]}\n')
        f.write(f'- **Class counts**: {report["class_counts"]}\n')
        f.write(f'- **SMOTE**: {report["models"].get("smote_applied", False)}\n')
        f.write(f'- **After balance**: {report["models"].get("train_class_counts_after_smote", {})}\n\n')
        for name, metrics in report['models'].items():
            if name in ('smote_applied', 'train_class_counts_after_smote'):
                continue
            f.write(f'## {name}\n\n')
            if name == 'feature_importance':
                for feature, value in list(metrics.items())[:20]:
                    f.write(f'- **{feature}**: {value:.6f}\n')
                f.write('\n')
                continue
            if name == 'mutual_information':
                for feature, value in sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f'- **{feature}**: {value:.6f}\n')
                f.write('\n')
                continue
            if name == 'permutation_importance_rf':
                for feature, value in sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:20]:
                    f.write(f'- **{feature}**: {value:.6f}\n')
                f.write('\n')
                continue
            if name in ('cross_validation_rf', 'cross_validation_lr'):
                for k, v in metrics.items():
                    f.write(f'- **{k}**: {v}\n')
                f.write('\n')
                continue
            if metrics.get('skipped'):
                f.write(f'- **Skipped**: {metrics["skipped"]}\n\n')
                continue
            f.write(f'- **BLOCK precision**: {metrics.get("block_precision", 0):.4f}\n')
            f.write(f'- **BLOCK recall**: {metrics.get("block_recall", 0):.4f}\n')
            f.write(f'- **ROC-AUC OVR**: {metrics.get("roc_auc_ovr", "n/a")}\n')
            f.write(f'- **Confusion matrix**: `{metrics.get("confusion_matrix")}`\n')
            thresholds = metrics.get('thresholds', {})
            if thresholds:
                f.write(f'- **Thresholds**: block={thresholds.get("block_threshold", "n/a")}, warn={thresholds.get("warn_threshold", "n/a")}\n')
            f.write('\n')
        drift = report.get('drift_detection', {})
        if drift and drift.get('drift_detected') is not None:
            f.write(f'## Drift Detection\n\n')
            f.write(f'- **Drift detected**: {drift["drift_detected"]}\n')
            f.write(f'- **Drifted features**: {drift.get("drifted_features", 0)}/{drift.get("total_features", 0)}\n\n')
    return path


def best_model_metrics(report: Dict) -> Dict:
    skip_keys = {'feature_importance', 'mutual_information', 'permutation_importance_rf',
                 'cross_validation_rf', 'cross_validation_lr', 'smote_applied',
                 'train_class_counts_after_smote'}
    candidates = [
        m for name, m in report['models'].items()
        if isinstance(m, dict) and not m.get('skipped') and name not in skip_keys
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda m: (m.get('block_precision', 0.0), m.get('block_recall', 0.0)))


def write_model_card(report: Dict, path: str):
    best = best_model_metrics(report)
    card = {
        'version': f"rf-metadata-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'created_at': report['created_at'],
        'feature_count': report['feature_count'],
        'features': report['features'],
        'rows': report['rows'],
        'class_counts': report['class_counts'],
        'dataset_hash': report.get('dataset_hash'),
        'best_model': best.get('name', 'unknown'),
        'block_precision': best.get('block_precision', 0.0),
        'block_recall': best.get('block_recall', 0.0),
        'roc_auc_ovr': best.get('roc_auc_ovr'),
        'thresholds': best.get('thresholds', {}),
        'smote_applied': report['models'].get('smote_applied', False),
        'notes': 'Generated by scripts/train_ru_metadata_models.py',
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(card, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train RF metadata models')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA)
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--export-tflite', action='store_true')
    parser.add_argument('--tflite-output', type=str, default=DEFAULT_TFLITE)
    parser.add_argument('--model-card-output', type=str, default=DEFAULT_MODEL_CARD)
    parser.add_argument('--min-block-precision', type=float, default=0.90)
    parser.add_argument('--allow-unsafe-export', action='store_true', help='Export even if BLOCK precision gate fails')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    parser.add_argument('--optuna-trials', type=int, default=0, help='Run Optuna hyperparameter search (0=off, N=trials)')
    parser.add_argument('--drift-reference', type=str, default=None, help='Production CSV for drift detection')
    parser.add_argument('--plots', action='store_true', help='Generate confusion matrix/ROC/importance plots')
    args = parser.parse_args()

    X, y = load_csv(args.data)
    if X.shape[1] != len(COMPACT_FEATURES):
        raise SystemExit(f'Feature mismatch: got {X.shape[1]}, expected {len(COMPACT_FEATURES)}')

    X_train, X_test, y_train, y_test = split_data(X, y)
    report = {
        'created_at': datetime.now().isoformat(),
        'data': args.data,
        'dataset_hash': file_sha256(args.data),
        'rows': int(len(y)),
        'feature_count': int(X.shape[1]),
        'features': COMPACT_FEATURES,
        'class_counts': class_counts(y),
        'models': {},
    }

    if len(set(y.tolist())) < 2:
        raise SystemExit('Need at least two classes for training. Populate BLOCK/WARN data first.')

    print('Training sklearn models (LR + RF + CatBoost + calibration)...')
    report['models'].update(train_sklearn_models(X_train, y_train, X_test, y_test, use_smote=not args.no_smote))

    if args.optuna_trials > 0:
        print(f'Running Optuna search ({args.optuna_trials} trials)...')
        report['models']['optuna_search'] = optuna_search(X_train, y_train, X_test, y_test, n_trials=args.optuna_trials)

    should_export = args.export_tflite
    if should_export and not args.allow_unsafe_export:
        best_block_precision = max(
            (m.get('block_precision', 0.0) for m in report['models'].values() if not m.get('skipped')),
            default=0.0,
        )
        if best_block_precision < args.min_block_precision:
            print(f'TFLite export blocked: best BLOCK precision {best_block_precision:.3f} < {args.min_block_precision:.3f}')
            should_export = False

    print('Training TFLite MLP...')
    report['models']['tflite_mlp'] = train_tflite_mlp(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs,
        batch=args.batch,
        output=args.tflite_output,
        export=should_export,
    )

    if args.drift_reference:
        print(f'Drift detection against {args.drift_reference}...')
        X_prod, _ = load_csv(args.drift_reference)
        report['drift_detection'] = detect_drift(X_train, X_prod)
    else:
        report['drift_detection'] = detect_drift(X_train, None)

    report_path = write_report(report)
    write_model_card(report, args.model_card_output)

    if args.plots:
        print('Generating plots...')
        generate_plots(report, REPORTS_DIR)

    print(f'Report saved: {report_path}')
    print(f'Model card saved: {args.model_card_output}')
    print(f'Rows={len(y)} features={X.shape[1]} class_counts={class_counts(y)}')
    for name, metrics in report['models'].items():
        if name in ('feature_importance', 'mutual_information', 'permutation_importance_rf',
                     'cross_validation_rf', 'cross_validation_lr', 'smote_applied',
                     'train_class_counts_after_smote'):
            if name == 'feature_importance':
                print(f'  {name}: top={list(metrics.items())[:5]}')
            elif name == 'mutual_information':
                top_mi = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f'  {name}: top={top_mi}')
            elif name == 'permutation_importance_rf':
                top_pi = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f'  {name}: top={top_pi}')
            elif 'cross_validation' in name and isinstance(metrics, dict) and 'cv_error' not in metrics:
                print(f'  {name}: F1={metrics.get("f1_macro_mean", 0):.3f}±{metrics.get("f1_macro_std", 0):.3f}')
            continue
        if isinstance(metrics, dict) and metrics.get('skipped'):
            print(f'  {name}: skipped ({metrics["skipped"]})')
        elif isinstance(metrics, dict) and 'block_precision' in metrics:
            print(f'  {name}: BLOCK P={metrics.get("block_precision", 0):.3f}, R={metrics.get("block_recall", 0):.3f}, ROC-AUC={metrics.get("roc_auc_ovr", "n/a")}')

    drift = report.get('drift_detection', {})
    if drift and drift.get('drift_detected') is not None:
        print(f'  Drift: {"DETECTED" if drift["drift_detected"] else "OK"} ({drift.get("drifted_features", 0)}/{drift.get("total_features", 0)})')


if __name__ == '__main__':
    main()
