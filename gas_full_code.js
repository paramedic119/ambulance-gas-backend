/**
 * Google Apps Script (GAS) の全体コードです。
 * 1. doGet: 地図表示時の軽量化（?type=map）
 * 2. doPost: ログ記録 ＋ データリセット（action: 'reset', password: '7710'）
 */

function doGet(e) {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = sheet.getDataRange().getValues();
    if (data.length <= 1) return ContentService.createTextOutput("[]").setMimeType(ContentService.MimeType.JSON);

    const headers = data[0];
    const rows = data.slice(1);
    const jsonData = rows.map(row => {
        let obj = {};
        headers.forEach((header, i) => obj[header] = row[i]);
        return obj;
    });

    // 地図解析時のみフィルタリングを実行
    if (e.parameter && e.parameter.type === 'map') {
        const uncomfTimes = jsonData
            .filter(d => parseInt(d.uncomfortable) === 1)
            .map(d => parseInt(d.time_ms));

        const WINDOW_MS = 30000;
        const filteredData = jsonData.filter((d, index) => {
            const t = parseInt(d.time_ms);
            const isNearUncomf = uncomfTimes.some(ut => t >= ut - WINDOW_MS && t <= ut + WINDOW_MS);
            return isNearUncomf || (index % 40 === 0);
        });
        return ContentService.createTextOutput(JSON.stringify(filteredData)).setMimeType(ContentService.MimeType.JSON);
    }

    return ContentService.createTextOutput(JSON.stringify(jsonData)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    let contents;
    try {
        contents = JSON.parse(e.postData.contents);
    } catch (err) {
        return ContentService.createTextOutput(JSON.stringify({ status: 'error', message: 'Invalid JSON' }))
            .setMimeType(ContentService.MimeType.JSON);
    }

    // --- データリセット機能 ---
    if (contents.action === 'reset') {
        if (contents.password === '7710') {
            const lastRow = sheet.getLastRow();
            if (lastRow > 1) {
                sheet.deleteRows(2, lastRow - 1);
            }
            return ContentService.createTextOutput(JSON.stringify({ status: 'success' }))
                .setMimeType(ContentService.MimeType.JSON);
        } else {
            return ContentService.createTextOutput(JSON.stringify({ status: 'error', message: 'Unauthorized' }))
                .setMimeType(ContentService.MimeType.JSON);
        }
    }

    // --- 通常のデータ記録 (配列で届く場合を想定) ---
    if (Array.isArray(contents) && contents.length > 0) {
        const lastCol = sheet.getLastColumn();
        if (lastCol === 0) return ContentService.createTextOutput(JSON.stringify({ status: 'error', message: 'No headers found' }))
            .setMimeType(ContentService.MimeType.JSON);

        const headers = sheet.getRange(1, 1, 1, lastCol).getValues()[0];
        const newRows = contents.map(log => {
            return headers.map(h => log[h] || "");
        });
        sheet.getRange(sheet.getLastRow() + 1, 1, newRows.length, headers.length).setValues(newRows);
    }

    return ContentService.createTextOutput(JSON.stringify({ status: 'success' }))
        .setMimeType(ContentService.MimeType.JSON);
}
