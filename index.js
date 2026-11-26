
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const express = require('express');

// ==========================================
// CONFIG
// ==========================================

const PORT = process.env.PORT || 3000;

const API_ID = process.env.API_ID || 'noname09';

const API_URL = process.env.API_URL ||
    'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?at=6256356b3f81d9bd5e0950ecd13c7df0&cl=R&cp=R&pf=web';

const DATA_FILE = path.join(
    process.env.DATA_DIR || __dirname,
    'lichsulc79md5.json'
);

const MAX_HISTORY = 200;
const CHECK_INTERVAL = 5000;

const app = express();

app.disable('x-powered-by');
app.set('json spaces', 2);

app.use((req, res, next) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
});

// ==========================================
// DATABASE
// ==========================================

function freshDatabase() {
    return {
        history: [],
        sequence: '',
        predictions: {},
        statistics: []
    };
}

function loadDatabase() {
    if (!fs.existsSync(DATA_FILE)) {
        return freshDatabase();
    }

    return {
        ...freshDatabase(),
        ...JSON.parse(
            fs.readFileSync(DATA_FILE, 'utf8')
        )
    };
}

let db = loadDatabase();

function saveDatabase() {
    db.history = db.history.slice(-MAX_HISTORY);

    db.sequence = db.history
        .map(item => item.result)
        .join('');

    db.statistics = db.statistics.slice(-200);

    fs.mkdirSync(
        path.dirname(DATA_FILE),
        { recursive: true }
    );

    const temporaryFile = DATA_FILE + '.tmp';

    fs.writeFileSync(
        temporaryFile,
        JSON.stringify(db, null, 2),
        'utf8'
    );

    fs.renameSync(temporaryFile, DATA_FILE);
}

// ==========================================
// STATISTICAL PREDICTION ENGINE
// ==========================================

// Bayesian smoothing
function bayesian(tai, total) {
    return (tai + 4) / (total + 8);
}

// Phân tích tần suất
function frequencyModel(sequence) {
    const tai = [...sequence]
        .filter(c => c === 't').length;

    return bayesian(tai, sequence.length);
}

// Markov / N-gram
function patternModel(sequence, length) {
    if (sequence.length <= length) {
        return 0.5;
    }

    const pattern = sequence.slice(-length);

    let tai = 0;
    let total = 0;

    for (
        let i = 0;
        i + length < sequence.length;
        i++
    ) {
        const historicalPattern = sequence.slice(
            i,
            i + length
        );

        if (historicalPattern === pattern) {
            total++;

            if (sequence[i + length] === 't') {
                tai++;
            }
        }
    }

    return bayesian(tai, total);
}

// Exponential weighted probability
function exponentialModel(sequence) {
    let tai = 4;
    let total = 8;

    [...sequence].reverse().forEach((c, i) => {
        const weight = Math.exp(-i / 30);

        total += weight;

        if (c === 't') {
            tai += weight;
        }
    });

    return tai / total;
}

// 9 mô hình thống kê
const MODELS = [
    () => 0.5,
    frequencyModel,
    s => patternModel(s, 1),
    s => patternModel(s, 2),
    s => patternModel(s, 3),
    s => patternModel(s, 4),
    s => patternModel(s, 5),
    s => patternModel(s, 6),
    exponentialModel
];

// Adaptive Ensemble + Walk Forward Validation
function predict(sequence) {
    const s = sequence.slice(-MAX_HISTORY);

    const losses = MODELS.map(() => 0);

    let count = 0;

    const start = Math.max(
        20,
        s.length - 80
    );

    for (let i = start; i < s.length; i++) {
        const previous = s.slice(0, i);
        const actual = s[i] === 't';

        MODELS.forEach((model, index) => {
            const p = model(previous);

            losses[index] -= Math.log(
                actual ? p : 1 - p
            );
        });

        count++;
    }

    // Điều chỉnh trọng số theo Log Loss
    const weights = losses.map(loss =>
        Math.exp(
            -8 * loss / Math.max(1, count)
        )
    );

    const totalWeight = weights.reduce(
        (a, b) => a + b,
        0
    );

    let probability = MODELS.reduce(
        (sum, model, index) => {
            return sum +
                model(s) * weights[index];
        },
        0
    ) / totalWeight;

    // Giảm độ tự tin khi dữ liệu chưa đủ
    const reliability = count / (count + 30);

    probability = 0.5 +
        (probability - 0.5) * reliability;

    const side = probability >= 0.5
        ? 't'
        : 'x';

    const confidence = Math.max(
        probability,
        1 - probability
    ) * 100;

    return {
        side,
        confidence,
        probabilityTai: probability
    };
}

// Lưu cố định dự đoán trước khi có kết quả
function getPrediction(sessionId) {
    const key = String(sessionId);

    if (!db.predictions[key]) {
        db.predictions[key] = predict(
            db.sequence
        );
    }

    return db.predictions[key];
}

// ==========================================
// API DATA PARSER
// ==========================================

function parseSessions(payload) {
    if (!Array.isArray(payload.list)) {
        throw new Error(
            'API schema error: expected list'
        );
    }

    const sessions = new Map();

    for (const item of payload.list) {
        const id = Number(item.id);

        const rawResult =
            item.resultTruyenThong;

        if (
            !Number.isSafeInteger(id) ||
            id <= 0
        ) {
            continue;
        }

        if (
            rawResult !== 'TAI' &&
            rawResult !== 'XIU'
        ) {
            continue;
        }

        const dices = Array.isArray(item.dices)
            ? item.dices.map(Number)
            : String(item.dices ?? '')
                .split(/[-,\s]+/)
                .map(Number);

        if (
            dices.length !== 3 ||
            !dices.every(d =>
                Number.isInteger(d) &&
                d >= 1 &&
                d <= 6
            )
        ) {
            continue;
        }

        const point = Number(item.point);

        const diceTotal = dices.reduce(
            (a, b) => a + b,
            0
        );

        if (point !== diceTotal) {
            continue;
        }

        sessions.set(id, {
            id,
            result: rawResult === 'TAI'
                ? 't'
                : 'x',
            dices,
            point
        });
    }

    return [...sessions.values()]
        .sort((a, b) => a.id - b.id);
}

// ==========================================
// FETCH API
// ==========================================

async function fetchSessions() {
    const response = await fetch(API_URL, {
        method: 'GET',
        headers: {
            'Accept': 'application/json'
        },
        signal: AbortSignal.timeout(4500)
    });

    if (!response.ok) {
        throw new Error(
            `API HTTP ${response.status}`
        );
    }

    const payload = await response.json();

    return parseSessions(payload);
}

// ==========================================
// TIME GMT+7
// ==========================================

function vietnamTime() {
    return new Date(
        Date.now() + 7 * 60 * 60 * 1000
    )
        .toISOString()
        .replace('Z', '+07:00');
}

function resultName(side) {
    return side === 't' ? 'tai' : 'xiu';
}

let lastSuccess = null;

// ==========================================
// CHECK NEW SESSION
// ==========================================

async function checkAPI() {
    const sessions = await fetchSessions();

    if (!sessions.length) {
        throw new Error(
            'Khong co phien hop le'
        );
    }

    // Lần đầu khởi động
    if (!db.history.length) {
        db.history = sessions.slice(
            -MAX_HISTORY
        );

        saveDatabase();

        const latest = db.history.at(-1);

        getPrediction(latest.id + 1);

        saveDatabase();

        console.log(
            `[INIT] Loaded ${db.history.length} sessions`
        );

        return;
    }

    const lastId = db.history.at(-1).id;

    const newSessions = sessions.filter(
        item => item.id > lastId
    );

    // Không có phiên mới: im lặng
    if (!newSessions.length) {
        return;
    }

    for (const item of newSessions) {
        const oldPrediction =
            db.predictions[String(item.id)];

        if (oldPrediction) {
            const correct =
                oldPrediction.side === item.result;

            db.statistics.push({
                id: item.id,
                predicted: oldPrediction.side,
                actual: item.result,
                correct,
                confidence: oldPrediction.confidence
            });

            console.log(
                `[CHECK] #${item.id} ` +
                (correct ? 'DUNG' : 'SAI')
            );
        }

        db.history.push(item);
    }

    const latest = newSessions.at(-1);

    // Xóa dự đoán đã hết hạn
    for (const key of Object.keys(
        db.predictions
    )) {
        if (Number(key) <= latest.id) {
            delete db.predictions[key];
        }
    }

    // Giới hạn lịch sử 200 phiên
    saveDatabase();

    // Dự đoán phiên tiếp theo
    const prediction = getPrediction(
        latest.id + 1
    );

    saveDatabase();

    console.log(
        '================================'
    );

    console.log(
        `[NEW] Phien: #${latest.id}`
    );

    console.log(
        `[RESULT] ${resultName(latest.result)}`
    );

    console.log(
        `[PREDICT] #${latest.id + 1}: ` +
        resultName(prediction.side)
    );

    console.log(
        `[CONFIDENCE] ` +
        prediction.confidence.toFixed(2) + '%'
    );

    console.log(
        `[HISTORY] ${db.history.length}/200`
    );

    console.log(
        '================================'
    );
}

// ==========================================
// JSON RESPONSE
// ==========================================

function getAPIResponse() {
    const latest = db.history.at(-1);

    if (!latest) {
        return null;
    }

    const prediction = getPrediction(
        latest.id + 1
    );

    return {
        id: API_ID,

        phienhientai: latest.id,

        ketqua: resultName(latest.result),

        xucxac: latest.dices,

        tong: latest.point,

        dudoanphien: latest.id + 1,

        dudoan: resultName(prediction.side),

        dotincay:
            prediction.confidence.toFixed(2) + '%',

        uptime: vietnamTime()
    };
}

// ==========================================
// EXPRESS ROUTES
// ==========================================

// API chính
app.get(['/', '/api'], (req, res) => {
    const data = getAPIResponse();

    if (!data) {
        return res.status(503).json({
            error: 'Dang cho du lieu API'
        });
    }

    return res.json(data);
});

// Thống kê 10 dự đoán gần nhất
app.get('/thongke', (req, res) => {
    const records = db.statistics.slice(-10);

    const wins = records.filter(
        item => item.correct
    ).length;

    const losses = records.length - wins;

    return res.json({
        id: API_ID,

        sophien: records.length,

        dung: wins,

        sai: losses,

        tyle: records.length
            ? (
                wins / records.length * 100
            ).toFixed(2) + '%'
            : null,

        chuoi: records.map(
            item => item.correct ? 'W' : 'L'
        ).join(''),

        chitiet: records.map(item => ({
            phien: item.id,

            dudoan: resultName(
                item.predicted
            ),

            ketqua: resultName(
                item.actual
            ),

            trangthai: item.correct
                ? 'DUNG'
                : 'SAI',

            dotincay:
                item.confidence.toFixed(2) + '%'
        }))
    });
});

// Lịch sử 200 phiên
app.get('/lichsu', (req, res) => {
    return res.json({
        id: API_ID,

        sophien: db.history.length,

        max: MAX_HISTORY,

        chuoi: db.sequence,

        history: db.history
    });
});

// Kiểm tra trạng thái
app.get('/health', (req, res) => {
    return res.json({
        status: lastSuccess
            ? 'online'
            : 'waiting',

        lastSuccess,

        history: db.history.length,

        uptime: process.uptime()
    });
});

// ==========================================
// AUTO LOOP EVERY 5 SECONDS
// ==========================================

function sleep(ms) {
    return new Promise(
        resolve => setTimeout(resolve, ms)
    );
}

async function autoLoop() {
    while (true) {
        const started = Date.now();

        try {
            await checkAPI();

            lastSuccess = vietnamTime();

        } catch (error) {
            console.error(
                '[API ERROR]',
                error.message
            );
        }

        const elapsed = Date.now() - started;

        await sleep(
            Math.max(
                0,
                CHECK_INTERVAL - elapsed
            )
        );
    }
}

// ==========================================
// START SERVER
// ==========================================

app.listen(PORT, '0.0.0.0', () => {
    console.log(
        '================================'
    );

    console.log(
        'LC79 MD5 API ONLINE'
    );

    console.log(
        `PORT: ${PORT}`
    );

    console.log(
        `API ID: ${API_ID}`
    );

    console.log(
        'CHECK INTERVAL: 5000ms'
    );

    console.log(
        'MAX HISTORY: 200'
    );

    console.log(
        'MODELS: 9'
    );

    console.log(
        '================================'
    );

    autoLoop();
});
