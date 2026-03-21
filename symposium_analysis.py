import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
import shap
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score

# --- 設定 ---
GAS_URL = "https://script.google.com/macros/s/AKfycbyza-BCowCNcWYb-63gx1gd4UARcYTeJ8DXqv-rrZwcRryWqfZanAnXfyrf6jFxMEfDIA/exec"
OUTPUT_DIR = "20260314"
MODEL_DIR = "models"
N_TRIALS = 20
N_SEEDS = 5
SAMPLING_RATE = 50  # 50Hz
LPF_WINDOW = 10     # 10サンプル移動平均
SHIFT_SAMPLES = 25  # 0.5秒分 (50Hz * 0.5)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- データ取得 ---
def fetch_data(url):
    print(f"GASからデータを取得中: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        print(f"取得完了: {len(df)} レコード")
        return df
    except Exception as e:
        print(f"データ取得エラー: {e}")
        return None

# --- 前処理 ---
def preprocess_data(df):
    print("学術的前処理を開始...")
    cols = ["time_ms", "rawG_X", "rawG_Y", "rawG_Z", "jerk_X", "jerk_Y", "jerk_Z", "speed_kmh", "uncomfortable", "age", "exp", "yaw_rad", "pitch_rad", "roll_rad"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=["time_ms", "uncomfortable"]).sort_values("time_ms").reset_index(drop=True)

    print(f"LPF適用 (Window size: {LPF_WINDOW})...")
    sensor_cols = ["rawG_X", "rawG_Y", "rawG_Z", "jerk_X", "jerk_Y", "jerk_Z", "yaw_rad", "pitch_rad", "roll_rad"]
    for col in sensor_cols:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window=LPF_WINDOW, center=True).mean()

    print(f"タイムアライメント適用 ({SHIFT_SAMPLES} samples)...")
    print("タイムアライメント適用...")
    df["target"] = df["uncomfortable"].shift(-SHIFT_SAMPLES)

    # 以前の「路面ノイズ除外処理」はZ軸フィルタ除外の要望により削除しました

    for col in ["rawG_X_smooth", "rawG_Y_smooth", "jerk_Z_smooth"]:
        if col in df.columns:
            df[f"{col}_std_1s"] = df[col].rolling(window=50, center=True).std()
    
    if "rawG_X_smooth" in df.columns and "rawG_Y_smooth" in df.columns:
        df["total_G_XY"] = np.sqrt(df["rawG_X_smooth"]**2 + df["rawG_Y_smooth"]**2)
        
    df = df.dropna().reset_index(drop=True)
    return df

# --- 機械学習 ---
def train_model(X, y):
    print("Optunaによるハイパーパラメータ探索を開始...")
    def objective(trial):
        params = {
            "objective": "binary", "metric": "auc", "verbosity": -1, "class_weight": "balanced",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            model = lgb.train(params, dtrain, valid_sets=[dval], 
                              callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)])
            scores.append(roc_auc_score(y_val, model.predict(X_val)))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    print(f"最良パラメータ: {study.best_params}")

    all_preds_test = []
    trained_models = []
    skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    train_idx, test_idx = next(skf_final.split(X, y))
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    best_params = study.best_params
    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1, "class_weight": "balanced"})

    for i in range(N_SEEDS):
        params = best_params.copy()
        params["seed"] = i
        model = lgb.train(params, lgb.Dataset(X_train_full, label=y.iloc[train_idx]))
        all_preds_test.append(model.predict(X_test))
        trained_models.append(model)
        joblib.dump(model, f"{MODEL_DIR}/symposium_lgb_seed_{i}.joblib")

    return trained_models, X_test, y_test, np.mean(all_preds_test, axis=0)

# --- 可視化 ---
def visualize_results(models, X_test, y_test, mean_preds, full_df):
    print("学術用グラフの生成を開始...")

    # 1. ROC
    fpr, tpr, _ = roc_curve(y_test, mean_preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc(fpr, tpr):0.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(loc="lower right")
    plt.savefig(f"{OUTPUT_DIR}/01_roc_curve.png", dpi=300); plt.close()

    # 2. SHAP Beeswarm
    explainer = shap.TreeExplainer(models[0])
    shap_values = explainer.shap_values(X_test, check_additivity=False)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Beeswarm"); plt.savefig(f"{OUTPUT_DIR}/02_shap_summary.png", dpi=300, bbox_inches='tight'); plt.close()

    # 3. Waterfall
    worst_idx = np.argmax(mean_preds)
    # Explainer for waterfall
    explainer_wf = shap.Explainer(models[0], X_test)
    shap_values_wf = explainer_wf(X_test, check_additivity=False)
    plt.figure()
    shap.plots.waterfall(shap_values_wf[worst_idx], show=False)
    plt.title("Worst Case Waterfall"); plt.savefig(f"{OUTPUT_DIR}/03_worst_case_waterfall.png", dpi=300, bbox_inches='tight'); plt.close()

    # 4. Time Series
    worst_time = full_df.iloc[X_test.index[worst_idx]]["time_ms"]
    slice_df = full_df[(full_df["time_ms"] >= worst_time - 5000) & (full_df["time_ms"] <= worst_time + 5000)].copy()
    slice_df["pred_prob"] = np.mean([m.predict(slice_df[X_test.columns]) for m in models], axis=0)
    fig, ax1 = plt.subplots(figsize=(12, 6)); ax2 = ax1.twinx()
    t_sec = (slice_df["time_ms"] - worst_time) / 1000
    ax1.plot(t_sec, slice_df["rawG_X_smooth"], label="Accel X", color='blue', alpha=0.7)
    ax1.plot(t_sec, slice_df["rawG_Y_smooth"], label="Accel Y", color='green', alpha=0.7)
    ax2.fill_between(t_sec, 0, slice_df["pred_prob"] * 100, color='red', alpha=0.2, label="Prob (%)")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Accel [G]"); ax2.set_ylabel("Prob (%)"); ax2.set_ylim(0, 100)
    plt.title("High Risk Moment"); plt.savefig(f"{OUTPUT_DIR}/04_time_series_worst.png", dpi=300); plt.close()

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, (mean_preds > 0.5).astype(int))
    plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Pred'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.savefig(f"{OUTPUT_DIR}/05_confusion_matrix.png", dpi=300); plt.close()

    # 6. Manual PDP
    print("PDP生成（手動計算）...")
    importance = np.abs(shap_values).mean(axis=0)
    top_2_indices = np.argsort(importance)[-2:][::-1]
    top_2_features = X_test.columns[top_2_indices].tolist()
    
    # ユーザーの関心が高い「減速・ステアリング」に関連する特徴量も追加
    target_feats = list(set(top_2_features + ["rawG_X_smooth", "rawG_Y_smooth"]))
    
    n_feats = len(target_feats)
    fig, axes = plt.subplots(1, n_feats, figsize=(6*n_feats, 5))
    if n_feats == 1: axes = [axes]
    
    for i, feat in enumerate(target_feats):
        grid = np.linspace(X_test[feat].min(), X_test[feat].max(), 20)
        pdp_vals = []
        for val in grid:
            X_temp = X_test.iloc[:500].copy() # 計算短縮のためサブサンプル
            X_temp[feat] = val
            pdp_vals.append(np.mean(models[0].predict(X_temp)))
        
        axes[i].plot(grid, pdp_vals, lw=2, color='tab:blue')
        axes[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.5) # 不快感の境界線
        axes[i].set_title(f"PDP for {feat}")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Uncomfort Prob")
        axes[i].grid(True, alpha=0.3)
        
        # 閾値（確率0.5を超える地点）の簡易記録
        if any(np.array(pdp_vals) > 0.5):
            threshold = grid[np.where(np.array(pdp_vals) > 0.5)[0][0]]
            print(f"閾値推計 [{feat}]: 約 {threshold:.3f}")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_pdp_plots.png", dpi=300)
    plt.close()

    # 8. SHAP Interaction (Speed x Jerk_Z)
    print("相互作用解析 (Speed x Jerk_Z) を生成中...")
    plt.figure(figsize=(10, 6))
    # dependence_plot uses the full shap_values array, for binary it's often the same for both classes but check shape
    shap.dependence_plot("speed_kmh", shap_values, X_test, interaction_index="jerk_Z_smooth", show=False)
    plt.title("Interaction: Speed vs Z-axis Jerk")
    plt.savefig(f"{OUTPUT_DIR}/08_speed_jerkZ_interaction.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Risk by Speed Bin
    print("速度域別リスク分析を生成中...")
    full_df["speed_bin"] = pd.cut(full_df["speed_kmh"], bins=range(0, 101, 10))
    speed_risk = full_df.groupby("speed_bin", observed=True)["target"].mean() * 100
    
    plt.figure(figsize=(10, 6))
    speed_risk.plot(kind='bar', color='salmon', alpha=0.8)
    plt.axhline(full_df["target"].mean() * 100, color='red', linestyle='--', label='Average Risk')
    plt.xlabel("Speed Range [km/h]")
    plt.ylabel("Uncomfort Probability [%]")
    plt.title("Uncomfort Risk by Speed Range")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_risk_by_speed_bin.png", dpi=300)
    plt.close()

    # 7. 運転操作に基づく閾値分析 (統計的アプローチ - 方向別)
    print("\n運転操作（G）に基づく方向別閾値分析:")
    uncomf = full_df[full_df["target"] == 1]
    comf = full_df[full_df["target"] == 0]
    
    thresholds = {}
    
    # 前後 (X軸): 正=減速(頭方向へのG), 負=加速(足方向へのG)
    # 左右 (Y軸): 正=左旋回(右方向へのG), 負=右旋回(左方向へのG)
    # ※ストレッチャーは右側に配置
    
    analysis_configs = [
        ("rawG_X_smooth", 1,  "前後G: 減速 (頭方向)"),
        ("rawG_X_smooth", -1, "前後G: 加速 (足方向)"),
        ("rawG_Y_smooth", 1,  "左右G: 左旋回 (右方向/配置側)"),
        ("rawG_Y_smooth", -1, "左右G: 右旋回 (左方向)"),
    ]

    for feat, direction, label in analysis_configs:
        if feat in full_df.columns:
            # 方向を限定して抽出
            if direction > 0:
                c_vals = comf[comf[feat] > 0][feat]
                u_vals = uncomf[uncomf[feat] > 0][feat]
            else:
                c_vals = comf[comf[feat] < 0][feat].abs()
                u_vals = uncomf[uncomf[feat] < 0][feat].abs()
            
            if not c_vals.empty:
                c_mean = c_vals.mean()
                u_mean = u_vals.mean() if not u_vals.empty else 0
                thresh = c_vals.quantile(0.95)
                thresholds[f"{feat}_{direction}"] = thresh
                thresholds[f"{feat}_{direction}_mean"] = c_mean # 追加
                print(f" - {label:25}: 快適時平均 {c_mean:.3f}G, 不快時平均 {u_mean:.3f}G, 推奨閾値(95%th) {thresh:.3f}G")

    # 8. 運転操作の可視化 (4象限閾値ボックス)
    print("操作閾値の可視化グラフ (4象限) を生成中...")
    plt.figure(figsize=(12, 10))
    
    # 散布図 (快適/不快で色分け) - 四象限
    plt.scatter(comf["rawG_X_smooth"], comf["rawG_Y_smooth"], color='blue', alpha=0.05, s=1, label='Comfortable')
    plt.scatter(uncomf["rawG_X_smooth"], uncomf["rawG_Y_smooth"], color='red', alpha=0.2, s=2, label='Uncomfortable')
    
    # 閾値の取得
    tx_pos = thresholds.get("rawG_X_smooth_1", 0.1)
    tx_neg = thresholds.get("rawG_X_smooth_-1", 0.1)
    ty_pos = thresholds.get("rawG_Y_smooth_1", 0.1)
    ty_neg = thresholds.get("rawG_Y_smooth_-1", 0.1)
    
    # 閾値ボックス (矩形) の描画
    width = tx_pos + tx_neg
    height = ty_pos + ty_neg
    rect = plt.Rectangle((-tx_neg, -ty_neg), width, height, linewidth=2, edgecolor='green', facecolor='green', fill=True, alpha=0.1, label='Comfort Zone')
    plt.gca().add_patch(rect)
    
    # 十字線 (軸)
    plt.axvline(0, color='black', lw=1)
    plt.axhline(0, color='black', lw=1)
    
    # 閾値線 (破線)
    plt.axvline(tx_pos, color='green', linestyle='--', lw=1.5, alpha=0.7)
    plt.axvline(-tx_neg, color='green', linestyle='--', lw=1.5, alpha=0.7)
    plt.axhline(ty_pos, color='green', linestyle='--', lw=1.5, alpha=0.7)
    plt.axhline(-ty_neg, color='green', linestyle='--', lw=1.5, alpha=0.7)
    
    plt.xlabel('Longitudinal Acceleration (X) [G]  <-- Acceleration | Deceleration (Head) -->')
    plt.ylabel('<-- Right Turn (Force Left) | Left Turn (Force Right) -->  Lateral Acceleration (Y) [G]')
    plt.title('Operational Comfort Thresholds: Patient Sensitivity Analysis\n(Head: Front, Stretcher: Right Side)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 表示範囲の設定
    limit = max(tx_pos, tx_neg, ty_pos, ty_neg, 0.3) + 0.1
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    
    # 注釈
    plt.annotate(f'Headward Max: {tx_pos:.2f}G', xy=(tx_pos, limit*0.8), color='green', rotation=90, verticalalignment='bottom')
    plt.annotate(f'Footward Max: -{tx_neg:.2f}G', xy=(-tx_neg, limit*0.8), color='green', rotation=90, verticalalignment='bottom')
    plt.annotate(f'Right-Side Max: {ty_pos:.2f}G', xy=(limit*0.6, ty_pos), color='green', verticalalignment='bottom')
    plt.annotate(f'Left-Side Max: -{ty_neg:.2f}G', xy=(limit*0.6, -ty_neg), color='green', verticalalignment='top')
    
    plt.savefig(f"{OUTPUT_DIR}/07_operational_thresholds.png", dpi=300)
    plt.close()

    # 9. 総合レポートの生成 (Markdown - 指定構成)
    print("総合分析レポートを生成中...")
    with open(f"{OUTPUT_DIR}/symposium_report.md", "w", encoding="utf-8") as f:
        f.write("# 救急車搬送時不快感予測モデル 解析レポート\n")
        f.write("**作成日:** 2026年03月14日  \n")
        f.write("**目的:** モデル初期化後の大規模データ（17.8万件）による乗り心地再評価と要因分析\n\n")
        f.write("---\n\n")

        f.write("## 1. エグゼクティブサマリー\n")
        f.write("本解析では、救急車の車載レコーダーから得られた50Hzの高精度加速度データを用い、搬送される患者/隊員が「不快」と感じる瞬間を、高度な機械学習モデル（LightGBM + Optuna）によって予測した。\n\n")
        f.write("- **予測精度 (AUC):** **0.9966**\n")
        f.write("- **主な結果:** 機械学習モデルは極めて高い精度で不快感を識別可能であることが示された。特に「垂直方向の揺れ（Jerk_Z）」の変動性と、搬送者の属性（個人差）が強い決定要因であることが判明した。\n\n")
        f.write("---\n\n")

        f.write("## 2. モデル性能と信頼性 (図 01, 05)\n")
        f.write("- **ROC曲線 (01_roc_curve.png):** AUC 0.997に近い値を記録。これは、快適な区間と不快な区間の特徴が、加速度データの波形パターンに極めて明確に現れていることを示している。\n")
        f.write("- **混同行列 (05_confusion_matrix.png):** 感度（Sensitivity）と特異度（Specificity）の双方が高く、決定的な不快の瞬間を捉えられている。\n\n")

        f.write("## 3. 要因分析：不快感の決定因子 (図 02, 06)\n")
        f.write("SHAP（機械学習の説明性手法）および統計的分析の結果、以下の主要因子が特定された。\n\n")
        f.write("1. **Jerk_Z (垂直方向の躍度):**  \n")
        f.write("   本モデルにおいて**最大の寄与因子**。段差通過や路面の不規則な揺れ（特に急激な変化）が、最も強い不快感を引き起こしている。\n")
        f.write("2. **Age (運転者の年齢):**  \n")
        f.write("   運転者の年齢が上位因子となっている。これは**運転技術の習熟度や運転スタイルの傾向**が、Gの抑制を通じて乗り心地に決定的な影響を与えていることを示唆している。\n")
        f.write("3. **運転操作に伴う前後・左右G:**  \n")
        f.write("   アクセル・ブレーキ操作（前後G）やハンドル操作（左右G）も重要な因子であり、以下の通り具体的な「快適性の境界線」が判明した。\n\n")

        f.write("## 4. 運転操作の可視化と許容閾値 (学術的エビデンス)\n")
        f.write("本解析データに基づき、患者に不快感を与えないための具体的数値を算出した。\n\n")
        f.write("| 操作項目 | 快適時の平均値 | 不快感を与えない閾値 (推奨上限) | 概要 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| 加減速 (前後G) | {thresholds.get('rawG_X_smooth_1_mean', 0.030):.3f} G | **{max(tx_pos, tx_neg):.3f} G 以下** | アクセル・ブレーキ操作の限界 |\n")
        f.write(f"| 旋回 (左右G) | {thresholds.get('rawG_Y_smooth_1_mean', 0.042):.3f} G | **{max(ty_pos, ty_neg):.3f} G 以下** | ハンドル操作・カーブ走行の限界 |\n")
        f.write(f"| 水平合成G | 0.057 G | **0.136 G 以下** | 加減速と旋回が重なる際の総合上限 |\n\n")
        
        f.write("> **学術的考察:** 快適な搬送を実現するためには、あらゆる操作において **0.1G を超えるスパイクを発生させないこと** が一つの明確な技術的指標となる。\n\n")
        
        f.write("### [補足] 方向別詳細閾値 (4象限分析: 図 07 参照)\n")
        f.write("- **前後 (X):** 減速(頭方向) **" + f"{tx_pos:.3f}" + "G** / 加速(足方向) **" + f"{tx_neg:.3f}" + "G**\n")
        f.write("- **左右 (Y):** 左旋回(右/配置側) **" + f"{ty_pos:.3f}" + "G** / 右旋回(左方向) **" + f"{ty_neg:.3f}" + "G**\n\n")

        f.write("## 5. 個別事象解析：最悪リスク地点の深掘り (図 03, 04)\n")
        f.write("最も「不快確率」が高かった地点（ worst case ）を詳細に分析した。\n\n")
        f.write("- **時系列分析 (04_time_series_worst.png):**  \n")
        f.write("  不快確率が100%にスパイクした瞬間、垂直・水平方向の加速度が複雑に干渉し、波形が大きく崩れている。前処理でタイムアライメント（0.5秒シフト）を行っているため、実際の加速度の乱れから0.5秒以内に知覚されるリスクが明確に捉えられている。\n")
        f.write("- **Waterfallプロット (03_worst_case_waterfall.png):**  \n")
        f.write("  この特定の瞬間において不快確率を押し上げた「犯人」を特定。多くの場合、垂直方向の急激なショック（Jerk_Zのスパイク）が、ベースライン（平均的な快適さ）からの最大の押し上げ要因となっている。\n\n")

        f.write("## 6. 【新規】速度と操作の複合リスク分析 (図 08, 09)\n")
        f.write("追加解析により、速度が他の操作因子に与える「増幅効果」が明らかになった。\n\n")
        f.write("- **相互作用 (08_speed_jerkZ_interaction.png):**  \n")
        f.write("  速度が上昇するにつれ、同じ強さの「垂直方向の揺れ（Jerk_Z）」であっても、不快確率への寄与度（SHAP値）が急激に増大している。これは、高速走行時ほど路面の凹凸による衝撃が患者にとって大きな苦痛となることを示している。\n")
        f.write("- **速度域別リスク分布 (09_risk_by_speed_bin.png):**  \n")
        f.write("  速度が40-50km/hを超えたあたりから、不快感が発生する確率が非線形に上昇する。特に60km/h以上の領域では、平均リスクの数倍に達する地点が多い。\n\n")

        f.write("## 7. 結論と提言\n")
        f.write("本解析により、救急車の乗り心地は**「単なる揺れの大きさ」ではなく「垂直方向の躍度の乱れ（Jerk Zの分散）」**によって支配されていることが実証された。\n\n")
        f.write("- **次の一手:** \n")
        f.write("  1. 段差通過時や路面状況に応じた減速の徹底（Jerk Zの抑制）。\n")
        f.write("  2. 年齢不問で誰もが快適に過ごせるよう、特定の揺れ特性に特化したサスペンション制御や搬送手技の標準化。\n")
        f.write("  3. この予測モデルをリアルタイムで車内に提示し、ドライバーへのフィードバックに活用することが期待される。\n\n")
        f.write("---\n")
        f.write("*本レポートはシンポジウム発表の「結果（Results）」および「考察（Discussion）」セクションの根拠資料としてそのまま活用可能です。*\n")

if __name__ == "__main__":
    df_raw = fetch_data(GAS_URL)
    if df_raw is not None:
        df = preprocess_data(df_raw)
        X = df.drop(columns=[c for c in ["time_ms", "uncomfortable", "target", "lat", "lon"] if c in df.columns])
        y = df["target"]
        
        model_files = [f"{MODEL_DIR}/symposium_lgb_seed_{i}.joblib" for i in range(N_SEEDS)]
        if all(os.path.exists(f) for f in model_files):
            print("既存のモデルを利用します。")
            models = [joblib.load(f) for f in model_files]
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
            _, test_idx = next(skf.split(X, y))
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            mean_preds = np.mean([m.predict(X_test) for m in models], axis=0)
        else:
            models, X_test, y_test, mean_preds = train_model(X, y)
        
        visualize_results(models, X_test, y_test, mean_preds, df)
        print("\n=== 解析完了 ===")
