# 救急車不快感解析：Google Colab での実行手順

作成した高度化解析プログラム [symposium_analysis_enhanced.py](file:///home/mihara/Desktop/運転アプリ/symposium_analysis_enhanced.py) を Google Colab で実行するための手順をまとめました。

## 準備するもの
- 作成した Python コード全体
- Google アカウント

## 実行ステップ

### 1. Google Colab を開く
[Google Colab](https://colab.research.google.com/) にアクセスし、「ノートブックを新規作成」をクリックします。

### 2. コードの貼り付け
最初のセルに [symposium_analysis_enhanced.py](file:///home/mihara/Desktop/運転アプリ/symposium_analysis_enhanced.py) の内容をすべてコピー＆ペーストします。

### 3. セルの実行
セルの左上にある「実行ボタン（▶）」をクリックします。

### 4. Google Drive へのアクセスの許可
実行途中で「Google ドライブへのアクセスを許可しますか？」というポップアップが表示されます。「Google ドライブに接続」を選択し、指示に従って許可してください。

### 5. 結果の確認
すべての処理が終わると、Google Drive 内の以下のフォルダに結果が自動保存されます。
- **場所:** `マイドライブ > ambulance_analysis > analysis_results_20260316`

## 生成される主な図表
| ファイル名 | 内容 | 初心者向けポイント |
| :--- | :--- | :--- |
| `02_shap_summary.png` | 要因分析 | ISO基準と現場のJerk、どちらが効いているか比較できます。 |
| `11_precision_recall_curve.png` | 判定精度 | AIがいかに「まれな不快イベント」を逃さないようにしたかを示します。 |
| `analysis_report_comprehensive.md` | 統合レポート | 解析結果を噛み砕いて説明した報告書です。 |

## 注意事項
- 初回実行時はライブラリのインストールに 1〜2 分ほど時間がかかる場合があります。
- Google Colab のランタイム接続が切れると計算も止まりますが、結果はすでに Drive に保存されています。
