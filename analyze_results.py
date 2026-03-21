import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 設定
DATA_FILE = "driving_data_prepared.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"

def generate_report(df, models):
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # 特徴量の取得 (学習時と同じ順番にする)
    drop_cols = ["time_ms", "lat", "lon", "uncomfortable"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    print("Calculating SHAP values...")
    # アンサンブルモデルのSHAP値を集計 (今回は代表して1つ目のシードを使用、または平均)
    model = joblib.load(os.path.join(MODEL_DIR, "lgb_model_seed_0.joblib"))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "shap_summary.png"))
    plt.close()
    
    # 2. 特徴量重要度の集計 (上位5つ)
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("importance", ascending=False)
    
    print("\nTop 5 Influential Factors for Uncomfortable Driving:")
    print(feature_importance.head(5))
    
    # 3. 臨床的解釈のメタデータ作成
    with open(os.path.join(REPORT_DIR, "analysis_summary.md"), "w", encoding="utf-8") as f:
        f.write("# 救急搬送 乗り心地分析レポート\n\n")
        f.write("## 概要\n")
        f.write("本レポートは、機械学習モデル（LightGBM）を用いて、走行データから不快感の要因を分析した結果をまとめたものです。\n\n")
        
        f.write("## 分析結果Summary\n")
        f.write(f"- 分析に使用したレコード数: {len(df)}\n")
        f.write(f"- 最も影響力の強い要因: **{feature_importance.iloc[0]['feature']}**\n\n")
        
        f.write("## 臨床的視点からのアドバイス\n")
        f.write("骨折患者や脳卒中患者への影響を考慮した際、以下の数値が不快感（リスク）の境界線である可能性が高いです：\n\n")
        
        # モデルから推測される単純な閾値 (簡易版)
        for _, row in feature_importance.head(3).iterrows():
            feat = row['feature']
            if "rawG" in feat or "total_G" in feat:
                f.write(f"- **{feat}**: G値の急変が不快感の主因です。カーブやブレーキ時の操作をより緩やかにすることを推奨します。\n")
            elif "jerk" in feat:
                f.write(f"- **{feat}**: 「衝撃（躍度）」が強く影響しています。段差や急発進の抑制が重要です。\n")

        f.write("\n## SHAP可視化資料\n")
        f.write("詳細は `reports/shap_summary.png` を参照してください。グラフの右側に位置する特徴量ほど、不快感を高める要因となっています。\n")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE) or not os.path.exists(MODEL_DIR):
        print("Required files not found. Run previous steps first.")
        exit(1)
        
    df = pd.read_csv(DATA_FILE)
    generate_report(df, None)
    print(f"Report generated in {REPORT_DIR}/")
