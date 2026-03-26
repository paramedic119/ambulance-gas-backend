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

    # --- 周波数自動判別と動的パラメータ設定 ---
    def process_ride(group):
        # サンプリング間隔(ms)の中央値からHzを推定
        dt_ms = group["time_ms"].diff().median()
        if pd.isna(dt_ms) or dt_ms <= 0:
            hz = 50.0  # デフォルト
        else:
            hz = 1000.0 / dt_ms
        
        # 3秒窓と5秒窓のサンプル数
        win_3s = max(1, int(3 * hz))
        win_5s = max(1, int(5 * hz))
        
        # 特徴量エンジニアリング
        if "total_G_XY" in group.columns:
            group["max_total_G_XY_3s"] = group["total_G_XY"].rolling(win_3s, min_periods=1).max()
        if "jerk_Z" in group.columns:
            group["max_jerk_Z_3s"] = group["jerk_Z"].rolling(win_3s, min_periods=1).max()
        if "rawG_Z" in group.columns:
            group["energy_Z_5s"] = (group["rawG_Z"]**2).rolling(win_5s, min_periods=1).mean()

        # 反応遅延ラベル (1.5秒前〜0.2秒前)
        # 1.5s = 1.5 * hz samples, 0.2s = 0.2 * hz samples
        if "uncomfortable" in group.columns:
            shift_start = int(1.5 * hz)
            shift_end = int(0.2 * hz)
            win_label = shift_start - shift_end
            # shift(-shift_start) で未来方向へずらし、window幅で過去(本来の未来)をカバー
            group["target"] = group["uncomfortable"].shift(-shift_start).rolling(window=win_label, min_periods=1).max().fillna(0)
        
        return group

    # 走行（ride_id）ごとに処理を適用
    df = df.groupby("ride_id", group_keys=False).apply(process_ride)
    
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
