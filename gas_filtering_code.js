/**
 * Google Apps Script (GAS) の doGet 関数を以下のように書き換えてください。
 * 大量データによるブラウザのフリーズを防ぎます。
 */
function doGet(e) {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = sheet.getDataRange().getValues();
    const headers = data[0];
    const rows = data.slice(1);

    // オブジェクトの配列に変換
    const jsonData = rows.map(row => {
        let obj = {};
        headers.forEach((header, i) => obj[header] = row[i]);
        return obj;
    });

    // --- フィルタリングロジック開始 ---

    // 地図解析ページ（type=map）からのリクエスト時のみフィルタリングを有効化
    if (e.parameter && e.parameter.type === 'map') {
        // 1. 不快ボタンが押された時間（time_ms）をすべて抽出
        const uncomfTimes = jsonData
            .filter(d => parseInt(d.uncomfortable) === 1)
            .map(d => parseInt(d.time_ms));

        const WINDOW_MS = 30000; // 前後30秒

        // 2. データの削減（間引き）
        const filteredData = jsonData.filter((d, index) => {
            const t = parseInt(d.time_ms);

            // 不快地点の前後30秒以内に含まれるか判定
            const isNearUncomf = uncomfTimes.some(ut => t >= ut - WINDOW_MS && t <= ut + WINDOW_MS);

            if (isNearUncomf) {
                return true; // 重要地点はすべて残す
            } else {
                // それ以外は40行に1行（20Hzなら2秒に1点）だけ残す（全体ルート把握用）
                return index % 40 === 0;
            }
        });

        return ContentService.createTextOutput(JSON.stringify(filteredData))
            .setMimeType(ContentService.MimeType.JSON);
    }

    // それ以外（統合分析など）は全データを返す
    return ContentService.createTextOutput(JSON.stringify(jsonData))
        .setMimeType(ContentService.MimeType.JSON);
}
