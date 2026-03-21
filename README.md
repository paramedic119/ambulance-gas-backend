# 救急搬送 乗り心地ロガー - GAS バックエンド

このリポジトリは、救急搬送時の乗り心地（加速度・躍度）データを収集・解析するための Google Apps Script (GAS) バックエンドコードを管理します。

## 含まれるコード

- `gas_full_code.js`: データの受信（POST）、保存、および統合分析用データ出力（GET）を担当。
- `gas_filtering_code.js`: 地図表示などのための軽量化フィルタリングロジック。

## 構成

- **データロガー**: [GitHub - datelogger](https://github.com/paramedic119/datelogger)
- **プロット・分析**: [GitHub - plot](https://github.com/paramedic119/plot)

## デプロイ方法

### GitHub Pages (Webアプリ)

`git push` を行うと自動的に GitHub Actions が走り、デプロイされます。
コマンドラインから手動でデプロイを実行する場合は以下を使用します：

```bash
npm run deploy:pages
```

### Google Apps Script (GAS)

以下のコマンドでデプロイ（反映）できます。
※内部的に `npx @google/clasp` を使用しており、初回実行時に Google アカウントへのログイン画面が開く場合があります。

```bash
npm run deploy:gas
```

### まとめてデプロイ

GitHub Pages と GAS の両方を同時にデプロイする場合：

```bash
npm run deploy
```

## Pythonによる高度な分析

スプレッドシートのデータを取得し、機械学習（LightGBM）を用いた詳細な分析を行います。

### セットアップ (仮想環境の作成)

現代の Linux (Ubuntu 等) では、システムの Python 破壊を防ぐため、仮想環境の使用が推奨されます。

```bash
# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化 (Linux)
source venv/bin/activate

# ライブラリのインストール
pip install -r requirements.txt
```

### 実行手順

仮想環境が有効な状態（ターミナルの先頭に `(venv)` と表示されている状態）で実行してください。

1. **データの取得と前処理**:

   ```bash
   python3 fetch_and_prepare.py
   ```

2. **モデルの学習と最適化 (Optuna + Seed Bagging)**:

    ```bash
    python3 train_ml_model.py
    ```

    - **一時停止と再開**:
      学習には時間がかかるため、途中で安全に停止して後で再開できる機能があります。
      - **一時停止**: ターミナルで `touch pause.flag` を実行してください。現在の試行（Trial）が終わり次第、安全に停止します。
      - **再開**: `pause.flag` ファイルを消去してから、再度 `python3 train_ml_model.py` を実行してください。自動的に続きから再開されます。

3. **分析レポートとSHAP可視化の生成**:

   ```bash
   python3 analyze_results.py
   ```

分析結果は `reports/` ディレクトリ内に生成されます。

## トラブルシューティング

### GitHub CLI (`gh`) のログインでエラーが出る場合

`gh auth login` でブラウザが開かない、またはエラーになる場合は、以下の手順を試してください：

1. `gh auth login` を実行。
2. `How would you like to authenticate GitHub CLI?` で `GitHub.com` を選択。
3. `What is your preferred protocol for Git operations?` で `HTTPS` を選択。
4. `Authenticate Git with your GitHub credentials?` で `Yes` を選択。
5. `How would you like to authenticate GitHub CLI?` で **`Paste an authentication token`** を選択。
6. GitHub の [Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens) で発行したトークン（スコープ: `workflow` が必要）を貼り付けてください。
