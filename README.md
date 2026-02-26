# 救急搬送 乗り心地ロガー - GAS バックエンド

このリポジトリは、救急搬送時の乗り心地（加速度・躍度）データを収集・解析するための Google Apps Script (GAS) バックエンドコードを管理します。

## 含まれるコード

- `gas_full_code.js`: データの受信（POST）、保存、および統合分析用データ出力（GET）を担当。
- `gas_filtering_code.js`: 地図表示などのための軽量化フィルタリングロジック。

## 構成

- **データロガー**: [GitHub - datelogger](https://github.com/paramedic119/datelogger)
- **プロット・分析**: [GitHub - plot](https://github.com/paramedic119/plot)

## 使い方

1. Google スプレッドシートを作成。
2. 「拡張機能」 > 「Apps Script」を開く。
3. `gas_full_code.js` の内容をコピー＆ペースト。
4. ウェブアプリとしてデプロイ（アクセス制限：全員）。
