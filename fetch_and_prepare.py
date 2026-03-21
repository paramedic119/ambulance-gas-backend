import requests
import pandas as pd
import numpy as np
import os

# GAS Web App URL
GAS_URL = "https://script.google.com/macros/s/AKfycbyza-BCowCNcWYb-63gx1gd4UARcYTeJ8DXqv-rrZwcRryWqfZanAnXfyrf6jFxMEfDIA/exec"

def fetch_data(url):
    print(f"Fetching data from {url}...")
    try:
        # GAS might redirect, requests handles this by default
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"Successfully fetched {len(data)} records.")
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(df):
    if df is None or df.empty:
        return None
    
    # 型変換
    numeric_cols = ["time_ms", "uncomfortable", "rawG_X", "rawG_Y", "rawG_Z", "jerk_X", "jerk_Y", "jerk_Z", "speed_kmh", "lat", "lon", "age", "exp"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 欠損値処理
    # (ここではまだuncomfortable=NaNを消さない。shiftで時間を引き戻すため)
    df = df.dropna(subset=["time_ms"])
    
    # --- 新規: 走行セッション（ride_id）の抽出 ---
    # 60秒以上のインターバルを別走行とみなす
    df["time_gap"] = df["time_ms"].diff().fillna(0)
    df["ride_id"] = (df["time_gap"] > 60000).cumsum()

    # 特徴量エンジニアリング (ラグ特徴量など)
    for col in ["rawG_X", "rawG_Y", "rawG_Z", "jerk_X", "jerk_Y", "jerk_Z"]:
        if col in df.columns:
            df[f"{col}_prev"] = df.groupby("ride_id")[col].shift(1)
            df[f"{col}_diff"] = df.groupby("ride_id")[col].diff()
            
    # 合成G
    if "rawG_X" in df.columns and "rawG_Y" in df.columns:
        df["total_G_XY"] = np.sqrt(df["rawG_X"]**2 + df["rawG_Y"]**2)

    # --- 新規: 累積ダメージ（時系列）特徴量の追加 ---
    # 過去3秒間(150 samples)のGの最大値
    if "total_G_XY" in df.columns:
        df["max_total_G_XY_3s"] = df.groupby("ride_id")["total_G_XY"].transform(lambda x: x.rolling(150, min_periods=1).max())
    if "jerk_Z" in df.columns:
        df["max_jerk_Z_3s"] = df.groupby("ride_id")["jerk_Z"].transform(lambda x: x.rolling(150, min_periods=1).max())
    
    # 過去5秒間(250 samples)のエネルギー（Z軸Gの2乗の移動平均）
    if "rawG_Z" in df.columns:
        df["energy_Z_5s"] = df.groupby("ride_id")["rawG_Z"].transform(lambda x: (x**2).rolling(250, min_periods=1).mean())

    # --- 新規: 反応遅延を吸収する区間ラベリング（ターゲットの再定義） ---
    # `uncomfortable = 1` の「1.5秒前〜0.2秒前」(10〜75 samples後) を正解とする
    if "uncomfortable" in df.columns:
        df["target"] = df.groupby("ride_id")["uncomfortable"].transform(
            lambda x: x.shift(-75).rolling(window=66, min_periods=1).max().fillna(0)
        )
        # 以降の学習用に不要な生ラベル等を描画用以外は消すが、ここでは出力に含める
    
    return df.dropna().reset_index(drop=True)

if __name__ == "__main__":
    df_raw = fetch_data(GAS_URL)
    if df_raw is not None:
        df_prepared = prepare_data(df_raw)
        if df_prepared is not None:
            # 保存
            output_file = "driving_data_prepared.csv"
            df_prepared.to_csv(output_file, index=False)
            print(f"Data prepared and saved to {output_file}")
            print(df_prepared.head())
        else:
            print("Data preparation failed.")
    else:
        print("Data fetch failed. Please check the GAS URL and internet connection.")
