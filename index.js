
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const express = require('express');

// ==========================================
// CONFIGURATION
// ==========================================

const PORT = process.env.PORT || 3000;
const API_ID = process.env.API_ID || 'noname09';

const API_URL = process.env.API_URL ||
  'https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?at=6256356b3f81d9bd5e0950ecd13c7df0&cl=R&cp=R&pf=web';

const FILE = path.join(
  process.env.DATA_DIR || __dirname,
  'lichsulc79md5.json'
);

const MAX = 200;
const INTERVAL = 5000;

const app = express();

app.disable('x-powered-by');
app.set('json spaces', 2);

app.use((req, res, next) => {
  res.set('Cache-Control', 'no-store');
  res.set('Access-Control-Allow-Origin', '*');
  next();
});

// ==========================================
// UTILITIES
// ==========================================

const vnTime = () =>
  new Date(Date.now() + 25200000)
    .toISOString()
    .replace('Z', '+07:00');

const name = s => s === 't' ? 'tai' : 'xiu';

const pct = p => p.toFixed(2) + '%';

const sequence = h =>
  h.map(r => r.result).join('');

const empty = () => ({
  history: [],
  sequence: '',
  predictions: {},
  statistics: []
});

// ==========================================
// DATABASE ENGINE
// ==========================================

function load() {
  if (!fs.existsSync(FILE)) {
    return empty();
  }

  const data = {
    ...empty(),
    ...JSON.parse(
      fs.readFileSync(FILE, 'utf8')
    )
  };

  if (
    !Array.isArray(data.history) ||
    !Array.isArray(data.statistics) ||
    !data.predictions ||
    typeof data.predictions !== 'object'
  ) {
    throw new Error('Invalid history file');
  }

  data.history = data.history.slice(-MAX);
  data.sequence = sequence(data.history);

  return data;
}

let db = load();

function commit(next) {
  next.history = next.history.slice(-MAX);
  next.sequence = sequence(next.history);
  next.statistics = next.statistics.slice(-MAX);

  fs.mkdirSync(
    path.dirname(FILE),
    { recursive: true }
  );

  fs.writeFileSync(
    FILE + '.tmp',
    JSON.stringify(next, null, 2),
    'utf8'
  );

  fs.renameSync(FILE + '.tmp', FILE);

  db = next;
}

// Không phân tích mẫu chuỗi qua các phiên bị thiếu
function contiguous(h) {
  let i = h.length - 1;

  while (
    i > 0 &&
    h[i].id === h[i - 1].id + 1
  ) {
    i--;
  }

  return sequence(
    h.slice(Math.max(0, i))
  );
}

// ==========================================
// ADAPTIVE STATISTICAL ENGINE
// ==========================================

const bayes = (t, n, a = 3) =>
  (t + a) / (n + 2 * a);

const clip = p =>
  Math.max(0.001, Math.min(0.999, p));

// MODEL 1: BAYESIAN FREQUENCY

function frequency(s) {
  return bayes(
    [...s].filter(c => c === 't').length,
    s.length
  );
}

// MODEL 2: EXPONENTIAL DECAY

function decay(s, halfLife) {
  let t = 0;
  let n = 0;

  for (let i = 0; i < s.length; i++) {
    const w = Math.pow(
      0.5,
      (s.length - 1 - i) / halfLife
    );

    n += w;

    if (s[i] === 't') {
      t += w;
    }
  }

  return bayes(t, n);
}

// MODEL 3: VARIABLE-ORDER MARKOV + N-GRAM

function ngram(s, k) {
  if (s.length <= k) {
    return 0.5;
  }

  const pattern = s.slice(-k);

  let t = 0;
  let n = 0;

  for (let i = 0; i + k < s.length; i++) {
    if (s.slice(i, i + k) !== pattern) {
      continue;
    }

    n++;

    if (s[i + k] === 't') {
      t++;
    }
  }

  return bayes(t, n);
}

// MODEL 4: STREAK ANALYSIS

function streak(s) {
  if (s.length < 2) {
    return 0.5;
  }

  let run = 1;

  for (
    let i = s.length - 2;
    i >= 0 && s[i] === s.at(-1);
    i--
  ) {
    run++;
  }

  let t = 0;
  let n = 0;
  let r = 0;

  for (let i = 0; i < s.length - 1; i++) {
    r = i > 0 && s[i] === s[i - 1]
      ? r + 1
      : 1;

    if (
      s[i] !== s.at(-1) ||
      Math.min(r, 6) !== Math.min(run, 6)
    ) {
      continue;
    }

    n++;

    if (s[i + 1] === 't') {
      t++;
    }
  }

  return bayes(t, n);
}

// MODEL 5: HAMMING SIMILARITY

function similarity(s) {
  const k = Math.min(
    8,
    Math.floor(s.length / 3)
  );

  if (k < 2) {
    return 0.5;
  }

  const pattern = s.slice(-k);

  let t = 0;
  let n = 0;

  for (let i = 0; i + k < s.length; i++) {
    let distance = 0;

    for (let j = 0; j < k; j++) {
      if (pattern[j] !== s[i + j]) {
        distance++;
      }
    }

    const w = Math.exp(-1.5 * distance);

    n += w;

    if (s[i + k] === 't') {
      t += w;
    }
  }

  return bayes(t, n);
}

// ==========================================
// 14 STATISTICAL MODEL CONFIGURATIONS
// ==========================================

const models = [
  () => 0.5,
  frequency,

  s => decay(s, 8),
  s => decay(s, 24),

  ...Array.from(
    { length: 8 },
    (_, i) => s => ngram(s, i + 1)
  ),

  streak,
  similarity
];

// ==========================================
// ADAPTIVE ENSEMBLE
// WALK FORWARD VALIDATION
// ==========================================

function predict(input) {
  const s = input.slice(-MAX);

  const losses = models.map(() => 0);

  let total = 0;

  // Kiểm nghiệm trên tối đa 100 phiên.
  // Không sử dụng dữ liệu tương lai.

  for (
    let i = Math.max(6, s.length - 100);
    i < s.length;
    i++
  ) {
    const prefix = s.slice(0, i);

    const actual = s[i] === 't';

    const w = Math.pow(
      0.5,
      (s.length - 1 - i) / 40
    );

    models.forEach((model, j) => {
      const p = clip(model(prefix));

      losses[j] -= w * Math.log(
        actual ? p : 1 - p
      );
    });

    total += w;
  }

  // Tự điều chỉnh trọng số theo Log Loss

  const weights = losses.map(
    (loss, j) =>
      (j === 0 ? 3 : 1) *
      Math.exp(
        -8 * (loss - losses[0]) /
        (total + 8)
      )
  );

  const sum = weights.reduce(
    (a, b) => a + b,
    0
  );

  // Kết hợp tất cả mô hình

  const raw = models.reduce(
    (a, model, j) =>
      a + weights[j] * model(s),
    0
  ) / sum;

  // Đánh giá mức cải thiện so với mốc 50%

  const bestGain = Math.max(
    0,
    losses[0] - Math.min(...losses)
  );

  const reliability =
    (s.length / (s.length + 24)) *
    (
      0.25 +
      0.75 * bestGain / (bestGain + 3)
    );

  // Confidence Shrinkage:
  // Hạn chế độ tự tin thiếu căn cứ.

  const p = clip(
    0.5 + (raw - 0.5) * reliability
  );

  return {
    side: p >= 0.5 ? 't' : 'x',

    confidence:
      100 * Math.max(p, 1 - p),

    probabilityTai: p
  };
}

// ==========================================
// SOURCE API PARSER
// ==========================================

function parse(payload) {
  if (!Array.isArray(payload.list)) {
    throw new Error(
      'API schema: missing list'
    );
  }

  const map = new Map();

  for (const r of payload.list) {
    const id = Number(r.id);

    const result = r.resultTruyenThong;

    if (
      !Number.isSafeInteger(id) ||
      id <= 0 ||
      !['TAI', 'XIU'].includes(result)
    ) {
      continue;
    }

    const dices = (
      Array.isArray(r.dices)
        ? r.dices
        : String(r.dices ?? '')
            .split(/[-,\s]+/)
    ).map(Number);

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

    const point = Number(r.point);

    if (
      point !== dices.reduce(
        (a, b) => a + b,
        0
      )
    ) {
      continue;
    }

    map.set(id, {
      id,
      result: result === 'TAI'
        ? 't'
        : 'x',
      dices,
      point
    });
  }

  return [...map.values()].sort(
    (a, b) => a.id - b.id
  );
}

// ==========================================
// NEW SESSION PROCESSING
// ==========================================

function apply(sessions) {
  const last =
    db.history.at(-1)?.id || 0;

  const incoming = sessions.filter(
    r => r.id > last
  );

  // Không có kết quả mới: im lặng

  if (
    !incoming.length &&
    (
      !last ||
      db.predictions[String(last + 1)]
    )
  ) {
    return false;
  }

  const next = {
    ...db,
    history: [...db.history],
    predictions: { ...db.predictions },
    statistics: [...db.statistics]
  };

  for (const r of incoming) {
    const old =
      next.predictions[String(r.id)];

    // Chỉ chấm dự đoán đã được lưu từ trước

    if (old) {
      next.statistics.push({
        id: r.id,
        predicted: old.side,
        actual: r.result,
        confidence: old.confidence,
        correct: old.side === r.result
      });
    }

    next.history.push(r);
  }

  // FIFO: giới hạn 200 phiên

  next.history = next.history.slice(-MAX);

  const latest = next.history.at(-1).id;

  for (const key of Object.keys(
    next.predictions
  )) {
    if (Number(key) <= latest) {
      delete next.predictions[key];
    }
  }

  const key = String(latest + 1);

  // Cố định dự đoán cho phiên tiếp theo

  if (!next.predictions[key]) {
    next.predictions[key] = predict(
      contiguous(next.history)
    );
  }

  commit(next);

  if (incoming.length) {
    console.log(
      '[NEW]',
      latest,
      name(next.predictions[key].side),
      pct(next.predictions[key].confidence)
    );
  }

  return true;
}

// ==========================================
// AUTO FETCH SOURCE API
// ==========================================

let lastSuccess = 0;
let lastError = false;

async function poll() {
  try {
    const res = await fetch(API_URL, {
      headers: {
        Accept: 'application/json'
      },

      signal: AbortSignal.timeout(4500)
    });

    if (!res.ok) {
      throw new Error(
        'HTTP ' + res.status
      );
    }

    const sessions = parse(
      await res.json()
    );

    if (!sessions.length) {
      throw new Error(
        'No valid sessions'
      );
    }

    apply(sessions);

    lastSuccess = Date.now();
    lastError = false;

  } catch (e) {
    lastError = true;

    console.error(
      '[POLL ERROR]',
      e.message
    );
  }
}

// ==========================================
// API OUTPUT
// ==========================================

function output() {
  const r = db.history.at(-1);

  const p = r &&
    db.predictions[String(r.id + 1)];

  if (!r || !p) {
    return null;
  }

  return {
    id: API_ID,
    phienhientai: r.id,
    ketqua: name(r.result),
    xucxac: r.dices,
    tong: r.point,
    dudoanphien: r.id + 1,
    dudoan: name(p.side),
    dotincay: pct(p.confidence),
    uptime: vnTime()
  };
}

// ==========================================
// CUSTOM BEAUTIFUL JSON FORMAT
// ==========================================

// Giữ mảng xúc xắc trên cùng một dòng:
// "xucxac": [6, 4, 4]

function pretty(data) {
  return '{\n' +

    Object.entries(data).map(
      ([k, v]) =>
        '  ' +
        JSON.stringify(k) +
        ': ' +

        (
          Array.isArray(v)

            ? '[' +
              v.map(x =>
                JSON.stringify(x)
              ).join(', ') +
              ']'

            : JSON.stringify(v)
        )

    ).join(',\n') +

    '\n}';
}

// ==========================================
// MAIN API ROUTES
// ==========================================

app.get(['/', '/api'], (req, res) => {
  const data = output();

  if (!data) {
    return res.status(503).json({
      error: 'Dang cho du lieu API'
    });
  }

  return res
    .type('application/json')
    .send(pretty(data));
});

// ==========================================
// HISTORY ENDPOINT
// ==========================================

app.get('/lichsu', (req, res) => {
  return res.json({
    id: API_ID,
    sophien: db.history.length,
    max: MAX,
    chuoi: db.sequence,
    history: db.history
  });
});

// ==========================================
// STATISTICS ENDPOINT
// ==========================================

app.get('/thongke', (req, res) => {
  const rows = db.statistics.slice(-10);

  const wins = rows.filter(
    r => r.correct
  ).length;

  return res.json({
    id: API_ID,

    sophien: rows.length,

    dung: wins,

    sai: rows.length - wins,

    tyle: rows.length
      ? pct(100 * wins / rows.length)
      : null,

    chuoi: rows.map(
      r => r.correct ? 'W' : 'L'
    ).join(''),

    chitiet: rows.map(r => ({
      phien: r.id,

      dudoan: name(r.predicted),

      ketqua: name(r.actual),

      trangthai: r.correct
        ? 'DUNG'
        : 'SAI',

      dotincay: pct(r.confidence)
    }))
  });
});

// ==========================================
// SERVER HEALTH
// ==========================================

app.get('/health', (req, res) => {
  return res.json({
    status:
      lastError ||
      (
        lastSuccess &&
        Date.now() - lastSuccess > 15000
      )

        ? 'degraded'

        : lastSuccess
          ? 'online'
          : 'waiting',

    lastSuccess: lastSuccess

      ? new Date(
          lastSuccess + 25200000
        ).toISOString()
         .replace('Z', '+07:00')

      : null,

    history: db.history.length,

    uptime: process.uptime()
  });
});

// ==========================================
// NON-OVERLAPPING AUTO LOOP
// ==========================================

async function loop() {
  const started = Date.now();

  await poll();

  setTimeout(
    loop,
    Math.max(
      0,
      INTERVAL - (Date.now() - started)
    )
  );
}

// ==========================================
// START SERVER
// ==========================================

if (require.main === module) {
  app.listen(PORT, '0.0.0.0', () => {
    console.log(
      '================================'
    );

    console.log('LC79 MD5 API ONLINE');
    console.log('PORT:', PORT);
    console.log('API ID:', API_ID);
    console.log('MODELS:', models.length);
    console.log('MAX HISTORY:', MAX);
    console.log('CHECK INTERVAL: 5s');
    console.log('RANDOM: DISABLED');

    console.log(
      '================================'
    );

    loop();
  });
}

module.exports = {
  predict,
  parse,
  pretty,
  apply,
  output
};
