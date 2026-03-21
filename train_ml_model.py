import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
import joblib
import os
import time

# 設定
DATA_FILE = "driving_data_prepared.csv"
MODEL_DIR = "models"
DB_FILE = "sqlite:///optuna_study.db"
STUDY_NAME = "driving_uncomfortable_analysis"
TARGET = "uncomfortable"
N_TRIALS = 100
N_SEEDS = 5
PAUSE_FLAG = "pause.flag"

def check_pause():
    if os.path.exists(PAUSE_FLAG):
        print(f"\n[PAUSE] '{PAUSE_FLAG}' を検知しました。処理を一時停止し、安全に終了します。")
        return True
    return False

def objective(trial, X, y, groups):
    # パラメータ提案
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gkf = GroupKFold(n_splits=5)
    scores = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=20)])
        
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)
        
    return np.mean(scores)

def pause_callback(study, trial):
    if check_pause():
        study.stop()

def train_final_models(X, y, best_params):
    print(f"Training final models with {N_SEEDS} seeds...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = []
    
    for seed in range(N_SEEDS):
        if check_pause():
            print("Final model training paused.")
            break
            
        model_path = os.path.join(MODEL_DIR, f"lgb_model_seed_{seed}.joblib")
        
        if os.path.exists(model_path):
            print(f"Seed {seed}: Model already exists. Skipping training.")
            models.append(joblib.load(model_path))
            continue
            
        print(f"Seed {seed}: Training started...")
        params = best_params.copy()
        params["random_state"] = seed
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        
        joblib.dump(model, model_path)
        models.append(model)
        
    return models

if __name__ == "__main__":
    if check_pause():
        print(f"警告: '{PAUSE_FLAG}' が存在するため開始できません。続行するにはこのファイルを削除してください。")
        exit(0)

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run fetch_and_prepare.py first.")
        exit(1)
        
    df = pd.read_csv(DATA_FILE)
    
    # 不要なカラムの除外 (ID系や緯度経度は一旦除外)
    # fetch_and_prepare.py で作った target を使うため、旧 uncomfortable は除外
    drop_cols = ["time_ms", "lat", "lon", "uncomfortable", "time_gap"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    if "ride_id" not in df.columns or "target" not in df.columns:
        print("Error: 'ride_id' or 'target' not found in data. Please run updated fetch_and_prepare.py.")
        exit(1)
        
    groups = df["ride_id"]
    y = df["target"]
    X = X.drop(columns=["ride_id", "target"])
    
    print(f"Features: {list(X.columns)}")
    print(f"Sample size: {len(df)}")
    print(f"ヒント: 途中で停止したい場合は `touch {PAUSE_FLAG}` を実行してください。")
    
    # Optuna with Storage (persistence)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_FILE,
        load_if_exists=True,
        direction="maximize"
    )
    
    remaining_trials = N_TRIALS - len(study.trials)
    if remaining_trials > 0:
        print(f"Resuming study. {len(study.trials)} trials completed. Running {remaining_trials} more.")
        study.optimize(lambda trial: objective(trial, X, y, groups), n_trials=remaining_trials, callbacks=[pause_callback])
    else:
        print(f"Optuna study already completed with {len(study.trials)} trials.")
    
    if len(study.trials) > 0 and not os.path.exists(PAUSE_FLAG):
        print("Best parameters:", study.best_params)
        # Final models
        train_final_models(X, y, study.best_params)
        print("All processes completed.")
    elif os.path.exists(PAUSE_FLAG):
        print(f"一時停止しました。再開するには {PAUSE_FLAG} を削除して再度実行してください。")
    else:
        print("No trials found. Check your data or parameters.")
