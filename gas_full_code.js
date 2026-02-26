/**
 * Google Apps Script (GAS) の全体コードです。
 * 1. doGet: 地図表示時の軽量化（?type=map）
 * 2. doPost: ログ記録 ＋ データリセット（action: 'reset', password: '7710'）
 */

function doGet(e) {
    const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    const data = sheet.getDataRange().getValues();
    if (data.length <= 1) return ContentService.createTextOutput("[]").setMimeType(ContentService.MimeType.JSON);

    const headers = data[0].map(h => h.toString().trim()); // Trim headers
    const rows = data.slice(1);

    // Filter out empty rows and map to objects
    const jsonData = rows.filter(row => row.join("").trim() !== "").map(row => {
        let obj = {};
        headers.forEach((header, i) => {
            let val = row[i];
            // Convert to number if possible for specific fields
            if (["time_ms", "uncomfortable", "rawG_X", "rawG_Y", "rawG_Z", "jerk_X", "jerk_Y", "jerk_Z", "speed_kmh", "lat", "lon", "age", "exp"].includes(header)) {
                val = parseFloat(val);
                if (isNaN(val)) val = 0;
            }
            obj[header] = val;
        });
        return obj;
    });

    // 地図解析時のみフィルタリングを実行
    if (e.parameter && e.parameter.type === 'map') {
        const uncomfTimes = jsonData
            .filter(d => d.uncomfortable === 1)
            .map(d => d.time_ms);

        const WINDOW_MS = 30000;
        const filteredData = jsonData.filter((d, index) => {
            const isNearUncomf = uncomfTimes.some(ut => d.time_ms >= ut - WINDOW_MS && d.time_ms <= ut + WINDOW_MS);
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
